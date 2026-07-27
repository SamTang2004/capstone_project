`timescale 1ns/1ps
// Concurrent Conv2D tile-group loader.
//
// The A, B0, and B1 im2col planners run independently. Their scalar DDR
// requests share one ordered port through a round-robin arbiter and response
// owner FIFO. Per-stream row packers emit each row as soon as its expected
// lanes retire, so packing and downstream bank fill overlap address planning.
module conv2d_tile_group_loader_int4_ddr #(
    parameter int TILE_SIZE = 8,
    parameter int OWNER_DEPTH = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        skip_weight_streams,
    input  logic [31:0] input_base_addr,
    input  logic [31:0] weight_base_addr,
    input  logic [15:0] input_c,
    input  logic [15:0] input_h,
    input  logic [15:0] input_w,
    input  logic [15:0] out_c,
    input  logic [15:0] out_w,
    input  logic [7:0]  kernel_h,
    input  logic [7:0]  kernel_w,
    input  logic [7:0]  stride_h,
    input  logic [7:0]  stride_w,
    input  logic [7:0]  pad_h,
    input  logic [7:0]  pad_w,
    input  logic [15:0] m0,
    input  logic [15:0] k0,
    input  logic [15:0] n0,
    input  logic [3:0]  m_len,
    input  logic [3:0]  k_len,
    input  logic [4:0]  n_cols,
    input  logic [1:0]  active_arrays,
    output logic        busy,
    output logic        done,
    output logic        error,
    output logic        read_req_valid,
    input  logic        read_req_ready,
    output logic [31:0] read_req_addr,
    output logic [4:0]  read_req_len,
    input  logic        read_data_valid,
    output logic        read_data_ready,
    input  logic [7:0]  read_data,
    input  logic        read_data_last,
    output logic        vector_valid,
    input  logic        vector_ready,
    output logic        vector_array,
    output logic        vector_matrix,
    output logic [2:0]  vector_row,
    output logic [7:0]  vector_mask,
    output logic [31:0] vector_data
);
    localparam int OWNER_PTR_W = $clog2(OWNER_DEPTH);

    logic active_r;
    logic [1:0] request_rr_r;
    logic [1:0] request_owner;
    logic request_found;
    logic request_capture;
    logic request_send;
    logic response_capture;
    logic response_consume;

    logic [1:0] owner_fifo [0:OWNER_DEPTH-1];
    logic [OWNER_PTR_W-1:0] owner_head_r, owner_tail_r;
    logic [OWNER_PTR_W:0] owner_count_r;
    logic [31:0] request_addr_fifo [0:1];
    logic [4:0] request_len_fifo [0:1];
    logic [1:0] request_owner_fifo [0:1];
    logic request_head_r, request_tail_r;
    logic [1:0] request_count_r;
    logic response_valid_r;
    logic [1:0] response_owner_r;
    logic [7:0] response_data_r;
    logic response_last_r;

    logic [2:0] loader_start;
    logic [2:0] loader_busy, loader_done, loader_error;
    logic [2:0] loader_req_valid, loader_req_ready;
    logic [31:0] loader_req_addr [0:2];
    logic [4:0] loader_req_len [0:2];
    logic [2:0] loader_data_valid, loader_data_ready;
    logic [2:0] loader_cache_we, loader_cache_matrix;
    logic [5:0] loader_cache_addr [0:2];
    logic signed [3:0] loader_cache_data [0:2];

    logic [2:0] pack_done, pack_error;
    logic [2:0] packed_valid, packed_ready;
    logic [2:0] packed_array, packed_matrix;
    logic [2:0] packed_row [0:2];
    logic [7:0] packed_mask [0:2];
    logic [31:0] packed_data [0:2];
    logic [2:0] stream_required_r;
    logic [2:0] stream_finished_r;

    function automatic logic [7:0] low_mask8(input logic [4:0] count);
        if (count >= 8)
            low_mask8 = 8'hff;
        else
            low_mask8 = (9'b1 << count) - 1'b1;
    endfunction

    wire [1:0] response_owner = owner_fifo[owner_head_r];
    wire all_finished =
        (stream_finished_r & stream_required_r) == stream_required_r;

    always_comb begin
        request_found = 1'b0;
        request_owner = request_rr_r;
        for (int offset = 0; offset < 3; offset++) begin
            int candidate;
            candidate = request_rr_r + offset;
            if (candidate >= 3)
                candidate = candidate - 3;
            if (!request_found && loader_req_valid[candidate]) begin
                request_found = 1'b1;
                request_owner = candidate[1:0];
            end
        end

        loader_req_ready = '0;
        if (active_r && request_found && request_count_r < 2)
            loader_req_ready[request_owner] = 1'b1;
    end

    assign request_capture =
        active_r && request_found && request_count_r < 2 &&
        loader_req_valid[request_owner];
    assign read_req_valid = request_count_r != 0 &&
                            owner_count_r < OWNER_DEPTH;
    assign read_req_addr = request_addr_fifo[request_head_r];
    assign read_req_len = request_len_fifo[request_head_r];
    assign request_send = read_req_valid && read_req_ready;
    assign read_data_ready = owner_count_r != 0 &&
                             (!response_valid_r ||
                              loader_data_ready[response_owner_r]);
    assign response_capture = read_data_valid && read_data_ready;
    assign response_consume = response_valid_r &&
                              loader_data_ready[response_owner_r];

    always_comb begin
        loader_data_valid = '0;
        if (response_valid_r)
            loader_data_valid[response_owner_r] = 1'b1;
    end

    always_comb begin
        vector_valid = 1'b0;
        vector_array = 1'b0;
        vector_matrix = 1'b0;
        vector_row = '0;
        vector_mask = '0;
        vector_data = '0;
        packed_ready = '0;
        for (int stream = 0; stream < 3; stream++) begin
            if (!vector_valid && packed_valid[stream]) begin
                vector_valid = 1'b1;
                vector_array = packed_array[stream];
                vector_matrix = packed_matrix[stream];
                vector_row = packed_row[stream];
                vector_mask = packed_mask[stream];
                vector_data = packed_data[stream];
                packed_ready[stream] = vector_ready;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active_r <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            request_rr_r <= '0;
            owner_head_r <= '0;
            owner_tail_r <= '0;
            owner_count_r <= '0;
            request_head_r <= 1'b0;
            request_tail_r <= 1'b0;
            request_count_r <= '0;
            response_valid_r <= 1'b0;
            response_owner_r <= '0;
            response_data_r <= '0;
            response_last_r <= 1'b0;
            loader_start <= '0;
            stream_required_r <= '0;
            stream_finished_r <= '0;
        end else begin
            done <= 1'b0;
            loader_start <= '0;

            if (start && !active_r) begin
                active_r <= 1'b1;
                error <= 1'b0;
                request_rr_r <= '0;
                owner_head_r <= '0;
                owner_tail_r <= '0;
                owner_count_r <= '0;
                request_head_r <= 1'b0;
                request_tail_r <= 1'b0;
                request_count_r <= '0;
                response_valid_r <= 1'b0;
                stream_required_r <= {
                    !skip_weight_streams &&
                        active_arrays == 2 && n_cols > 8,
                    !skip_weight_streams,
                    1'b1
                };
                stream_finished_r <= '0;
                loader_start[0] <= 1'b1;
                loader_start[1] <= !skip_weight_streams;
                loader_start[2] <= !skip_weight_streams &&
                                   active_arrays == 2 && n_cols > 8;
            end

            if (request_capture) begin
                request_addr_fifo[request_tail_r] <=
                    loader_req_addr[request_owner];
                request_len_fifo[request_tail_r] <=
                    loader_req_len[request_owner];
                request_owner_fifo[request_tail_r] <= request_owner;
                request_tail_r <= ~request_tail_r;
                request_rr_r <= request_owner == 2 ? 0 :
                                request_owner + 1'b1;
            end
            if (request_send) begin
                owner_fifo[owner_tail_r] <=
                    request_owner_fifo[request_head_r];
                owner_tail_r <= owner_tail_r + 1'b1;
                request_head_r <= ~request_head_r;
            end
            unique case ({request_capture, request_send})
                2'b10: request_count_r <= request_count_r + 1'b1;
                2'b01: request_count_r <= request_count_r - 1'b1;
                default: request_count_r <= request_count_r;
            endcase
            if (response_capture)
                owner_head_r <= owner_head_r + 1'b1;

            unique case ({request_send, response_capture})
                2'b10: owner_count_r <= owner_count_r + 1'b1;
                2'b01: owner_count_r <= owner_count_r - 1'b1;
                default: owner_count_r <= owner_count_r;
            endcase

            unique case ({response_capture, response_consume})
                2'b10: response_valid_r <= 1'b1;
                2'b01: response_valid_r <= 1'b0;
                2'b11: response_valid_r <= 1'b1;
                default: response_valid_r <= response_valid_r;
            endcase
            if (response_capture) begin
                response_owner_r <= response_owner;
                response_data_r <= read_data;
                response_last_r <= read_data_last;
            end

            if (!(start && !active_r)) begin
                for (int stream = 0; stream < 3; stream++) begin
                    // The row packer holds done until it samples this
                    // transaction's start pulse. Ignore both the parent
                    // restart edge and the following child restart edge.
                    if (pack_done[stream] && !loader_start[stream])
                        stream_finished_r[stream] <= 1'b1;
                end

                if (active_r &&
                    ((stream_finished_r |
                      (pack_done & ~loader_start)) & stream_required_r) ==
                     stream_required_r) begin
                    active_r <= 1'b0;
                    done <= 1'b1;
                end
            end

            if (|(loader_error | pack_error) ||
                (response_capture && !read_data_last))
                error <= 1'b1;
        end
    end

    assign busy = active_r;

    for (genvar stream = 0; stream < 3; stream++) begin : gen_loaders
        localparam logic IS_B = stream != 0;
        localparam logic IS_B1 = stream == 2;
        logic [3:0] b_cols;

        always_comb begin
            if (!IS_B)
                b_cols = k_len;
            else if (!IS_B1)
                b_cols = n_cols > 8 ? 4'd8 : n_cols[3:0];
            else
                b_cols = n_cols > 8 ? n_cols - 8 : 4'd0;
        end

        conv2d_tile_loader_int4_ddr #(.TILE_SIZE(TILE_SIZE)) loader (
            .clk, .rst_n, .start(loader_start[stream]),
            .matrix_select(IS_B),
            .input_base_addr, .weight_base_addr,
            .input_c, .input_h, .input_w, .out_c, .out_w,
            .kernel_h, .kernel_w, .stride_h, .stride_w, .pad_h, .pad_w,
            .m0, .k0,
            .n0(n0 + (IS_B1 ? 16'd8 : 16'd0)),
            .tile_rows(IS_B ? k_len : m_len),
            .tile_cols(b_cols),
            .busy(loader_busy[stream]), .done(loader_done[stream]),
            .error(loader_error[stream]),
            .read_req_valid(loader_req_valid[stream]),
            .read_req_ready(loader_req_ready[stream]),
            .read_req_addr(loader_req_addr[stream]),
            .read_req_len(loader_req_len[stream]),
            .read_data_valid(loader_data_valid[stream]),
            .read_data_ready(loader_data_ready[stream]),
            .read_data(response_data_r),
            .read_data_last(response_last_r),
            .cache_we(loader_cache_we[stream]),
            .cache_matrix(loader_cache_matrix[stream]),
            .cache_addr(loader_cache_addr[stream]),
            .cache_wdata(loader_cache_data[stream])
        );

        int4_streaming_scalar_row_packer packer (
            .clk, .rst_n,
            .start(loader_start[stream]),
            .finish(loader_done[stream]),
            .transaction_array(IS_B1),
            .transaction_matrix(IS_B),
            .expected_rows(IS_B ? k_len : m_len),
            .expected_lane_mask(
                IS_B ? low_mask8({1'b0, b_cols}) :
                       low_mask8({1'b0, k_len})
            ),
            .scalar_valid(loader_cache_we[stream]),
            .scalar_addr(loader_cache_addr[stream]),
            .scalar_data(loader_cache_data[stream]),
            .vector_valid(packed_valid[stream]),
            .vector_ready(packed_ready[stream]),
            .vector_array(packed_array[stream]),
            .vector_matrix(packed_matrix[stream]),
            .vector_row(packed_row[stream]),
            .vector_mask(packed_mask[stream]),
            .vector_data(packed_data[stream]),
            .done(pack_done[stream]),
            .error(pack_error[stream])
        );
    end
endmodule
