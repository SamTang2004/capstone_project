module packed_w4_weight_reader (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        start,
    input  logic [31:0] base_addr,
    input  logic [31:0] weight_count,
    output logic        busy,
    output logic        done,
    output logic        error,

    // Minimal request/response connection to memory_io_driver's qword port.
    output logic        read_req_valid,
    input  logic        read_req_ready,
    output logic [31:0] read_req_addr,
    output logic [6:0]  read_req_len,
    output logic        read_req_wide,
    output logic        read_req_qword,
    output logic        read_req_allow_ahead,
    input  logic        read_qword_valid,
    output logic        read_qword_ready,
    input  logic [63:0] read_qword_data,
    input  logic [7:0]  read_qword_keep,
    input  logic        read_qword_last,

    // Eight packed W4 weights per compute-vector transfer.
    output logic        weight_valid,
    input  logic        weight_ready,
    output logic [31:0] packed_weights,
    output logic [7:0]  weight_lane_valid,
    output logic        weight_last
);
    logic busy_r;
    logic [31:0] next_addr;
    logic [31:0] bytes_unrequested;
    logic [31:0] weights_remaining;

    // One qword is the complete local buffer: two 8-weight vectors.
    logic        word_valid;
    logic [63:0] word_data;
    logic [7:0]  word_keep;
    logic        upper_half;

    logic [6:0] bytes_to_boundary;
    logic [6:0] request_bytes;
    logic [3:0] selected_keep;
    logic [31:0] selected_data;
    logic       output_fire;
    logic       word_retires;

    always_comb begin
        bytes_to_boundary = 7'd64 - {1'b0, next_addr[5:0]};
        if (bytes_unrequested < bytes_to_boundary)
            request_bytes = bytes_unrequested[6:0];
        else
            request_bytes = bytes_to_boundary;
    end

    assign busy = busy_r;
    assign read_req_valid = busy_r && bytes_unrequested != 0;
    assign read_req_addr = next_addr;
    assign read_req_len = request_bytes;
    assign read_req_wide = 1'b1;
    assign read_req_qword = 1'b1;
    assign read_req_allow_ahead = 1'b1;

    assign selected_keep = upper_half ? word_keep[7:4] : word_keep[3:0];
    assign selected_data = upper_half ? word_data[63:32] : word_data[31:0];
    assign weight_valid = busy_r && word_valid && selected_keep != 0;
    assign weight_last = weight_valid && weights_remaining <= 8;
    assign output_fire = weight_valid && weight_ready;

    always_comb begin
        weight_lane_valid = '0;
        packed_weights = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (lane < weights_remaining && selected_keep[lane >> 1]) begin
                weight_lane_valid[lane] = 1'b1;
                packed_weights[lane*4 +: 4] = selected_data[lane*4 +: 4];
            end
        end
    end

    // Replace a retiring qword on the same edge, avoiding a bubble between
    // adjacent words while retaining only one 64-bit local storage entry.
    assign word_retires = output_fire &&
        (upper_half || word_keep[7:4] == 0 || weights_remaining <= 8);
    assign read_qword_ready = busy_r && (!word_valid || word_retires);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy_r <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            next_addr <= '0;
            bytes_unrequested <= '0;
            weights_remaining <= '0;
            word_valid <= 1'b0;
            word_data <= '0;
            word_keep <= '0;
            upper_half <= 1'b0;
        end else begin
            done <= 1'b0;

            if (!busy_r && start) begin
                error <= 1'b0;
                word_valid <= 1'b0;
                upper_half <= 1'b0;
                if (weight_count == 0 || base_addr[1:0] != 0) begin
                    error <= 1'b1;
                    done <= 1'b1;
                end else begin
                    busy_r <= 1'b1;
                    next_addr <= base_addr;
                    bytes_unrequested <= (weight_count + 1'b1) >> 1;
                    weights_remaining <= weight_count;
                end
            end

            if (read_req_valid && read_req_ready) begin
                next_addr <= next_addr + request_bytes;
                bytes_unrequested <= bytes_unrequested - request_bytes;
            end

            if (output_fire) begin
                if (weights_remaining <= 8) begin
                    weights_remaining <= '0;
                    word_valid <= 1'b0;
                    busy_r <= 1'b0;
                    done <= 1'b1;
                end else begin
                    weights_remaining <= weights_remaining - 8;
                    if (!upper_half && word_keep[7:4] != 0) begin
                        upper_half <= 1'b1;
                    end else begin
                        word_valid <= 1'b0;
                        upper_half <= 1'b0;
                    end
                end
            end

            if (read_qword_valid && read_qword_ready) begin
                word_valid <= 1'b1;
                word_data <= read_qword_data;
                word_keep <= read_qword_keep;
                upper_half <= 1'b0;
                if ((read_qword_last && read_qword_keep == 0) ||
                    (!read_qword_last && read_qword_keep != 8'hff))
                    error <= 1'b1;
            end
        end
    end
endmodule
