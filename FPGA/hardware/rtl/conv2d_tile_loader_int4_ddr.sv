`timescale 1ns/1ps
// Packed signed-INT4 direct im2col loader for one 8x8 GEMM tile.
//
// Address generation, DDR requests, and ordered tile emission are independent.
// An eight-entry circular reorder window matches the memory driver's queue.
// Returned bytes are tagged and retired in tile order, so variable DDR latency
// cannot perturb the array without requiring a 64-way associative issue mux.
module conv2d_tile_loader_int4_ddr #(
    parameter int TILE_SIZE = 8,
    parameter int READ_WINDOW = 8
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               matrix_select,
    input  logic [31:0]        input_base_addr,
    input  logic [31:0]        weight_base_addr,
    input  logic [15:0]        input_c,
    input  logic [15:0]        input_h,
    input  logic [15:0]        input_w,
    input  logic [15:0]        out_c,
    input  logic [15:0]        out_w,
    input  logic [7:0]         kernel_h,
    input  logic [7:0]         kernel_w,
    input  logic [7:0]         stride_h,
    input  logic [7:0]         stride_w,
    input  logic [7:0]         pad_h,
    input  logic [7:0]         pad_w,
    input  logic [15:0]        m0,
    input  logic [15:0]        k0,
    input  logic [15:0]        n0,
    input  logic [3:0]         tile_rows,
    input  logic [3:0]         tile_cols,
    output logic               busy,
    output logic               done,
    output logic               error,
    output logic               read_req_valid,
    input  logic               read_req_ready,
    output logic [31:0]        read_req_addr,
    output logic [4:0]         read_req_len,
    input  logic               read_data_valid,
    output logic               read_data_ready,
    input  logic [7:0]         read_data,
    input  logic               read_data_last,
    output logic               cache_we,
    output logic               cache_matrix,
    output logic [5:0]         cache_addr,
    output logic signed [3:0]  cache_wdata
);
    localparam int ROB_DEPTH = READ_WINDOW;
    localparam int ROB_PTR_W = $clog2(ROB_DEPTH);
    localparam int TAG_PTR_W = $clog2(READ_WINDOW);

    typedef enum logic [4:0] {
        IDLE, DECODE_M, DECODE_K_IC, DECODE_K_KH, PREPARE,
        COORD_MUL, COORD_ADD, CHECK_BOUNDS,
        ADDR_HORNER0, ADDR_HORNER1, ADDR_HORNER2,
        PACK_ADDRESS, RECORD, ADVANCE,
        WAIT_DRAIN, FAIL
    } state_t;

    state_t state;
    logic active_r, plan_done_r, matrix_r;
    logic [3:0] row_r, col_r, rows_r, cols_r;
    logic [6:0] generated_count_r, emitted_count_r, total_count_r;
    logic [31:0] input_base_r, weight_base_r;
    logic [15:0] in_c_r, in_h_r, in_w_r, out_c_r, out_w_r;
    logic [7:0] kh_size_r, kw_size_r, sh_r, sw_r, ph_r, pw_r;
    logic [15:0] n0_r;
    logic [15:0] m_remaining_r, k_remaining_r, kernel_area_r;
    logic [15:0] base_oh_r, base_ow_r, row_oh_r, row_ow_r;
    logic [15:0] base_ic_r, base_kh_r, base_kw_r;
    logic [15:0] k_ic_r, k_kh_r, k_kw_r;
    logic signed [16:0] input_y_r, input_x_r;
    logic [31:0] coord_h_product_r, coord_w_product_r;
    logic [31:0] horner0_r, horner1_r, logical_index_r;
    logic [31:0] byte_addr_r;
    logic nibble_select_r, pad_r;

    logic rob_valid [0:ROB_DEPTH-1];
    logic rob_issued [0:ROB_DEPTH-1];
    logic rob_ready [0:ROB_DEPTH-1];
    logic rob_nibble [0:ROB_DEPTH-1];
    logic [31:0] rob_addr [0:ROB_DEPTH-1];
    logic [5:0] rob_dest [0:ROB_DEPTH-1];
    logic [7:0] rob_data [0:ROB_DEPTH-1];

    logic [ROB_PTR_W-1:0] tag_fifo [0:READ_WINDOW-1];
    logic [TAG_PTR_W-1:0] tag_head_r, tag_tail_r;
    logic [TAG_PTR_W:0] tag_count_r;
    logic [6:0] issue_cursor_r;
    logic [ROB_PTR_W-1:0] issue_index;
    logic issue_found, issue_fire, response_fire, output_fire;

    assign issue_index = issue_cursor_r[ROB_PTR_W-1:0];
    assign issue_found = issue_cursor_r < generated_count_r &&
                         rob_valid[issue_index] &&
                         !rob_issued[issue_index] &&
                         !rob_ready[issue_index];

    assign busy = active_r;
    assign read_req_valid = active_r && issue_found &&
                            tag_count_r < READ_WINDOW;
    assign read_req_addr = rob_addr[issue_index];
    assign read_req_len = 5'd1;
    assign read_data_ready = tag_count_r != 0;
    assign issue_fire = read_req_valid && read_req_ready;
    assign response_fire = read_data_valid && read_data_ready;

    wire [ROB_PTR_W-1:0] emit_index =
        emitted_count_r[ROB_PTR_W-1:0];
    assign cache_we = active_r && emitted_count_r < generated_count_r &&
                      rob_valid[emit_index] &&
                      rob_ready[emit_index];
    assign cache_matrix = matrix_r;
    assign cache_addr = rob_dest[emit_index];
    assign cache_wdata = rob_nibble[emit_index] ?
                         $signed(rob_data[emit_index][7:4]) :
                         $signed(rob_data[emit_index][3:0]);
    assign output_fire = cache_we;

    task automatic increment_k;
        begin
            if (k_kw_r + 1 >= kw_size_r) begin
                k_kw_r <= '0;
                if (k_kh_r + 1 >= kh_size_r) begin
                    k_kh_r <= '0;
                    k_ic_r <= k_ic_r + 1'b1;
                end else begin
                    k_kh_r <= k_kh_r + 1'b1;
                end
            end else begin
                k_kw_r <= k_kw_r + 1'b1;
            end
        end
    endtask

    task automatic increment_spatial;
        begin
            if (row_ow_r + 1 >= out_w_r) begin
                row_ow_r <= '0;
                row_oh_r <= row_oh_r + 1'b1;
            end else begin
                row_ow_r <= row_ow_r + 1'b1;
            end
        end
    endtask

    integer entry;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            active_r <= 1'b0;
            plan_done_r <= 1'b0;
            matrix_r <= 1'b0;
            row_r <= '0; col_r <= '0; rows_r <= '0; cols_r <= '0;
            generated_count_r <= '0; emitted_count_r <= '0;
            total_count_r <= '0;
            input_base_r <= '0; weight_base_r <= '0;
            in_c_r <= '0; in_h_r <= '0; in_w_r <= '0;
            out_c_r <= '0; out_w_r <= '0;
            kh_size_r <= '0; kw_size_r <= '0;
            sh_r <= '0; sw_r <= '0; ph_r <= '0; pw_r <= '0;
            n0_r <= '0;
            m_remaining_r <= '0; k_remaining_r <= '0;
            kernel_area_r <= '0;
            base_oh_r <= '0; base_ow_r <= '0;
            row_oh_r <= '0; row_ow_r <= '0;
            base_ic_r <= '0; base_kh_r <= '0; base_kw_r <= '0;
            k_ic_r <= '0; k_kh_r <= '0; k_kw_r <= '0;
            input_y_r <= '0; input_x_r <= '0;
            coord_h_product_r <= '0; coord_w_product_r <= '0;
            horner0_r <= '0; horner1_r <= '0;
            logical_index_r <= '0; byte_addr_r <= '0;
            nibble_select_r <= 1'b0; pad_r <= 1'b0;
            tag_head_r <= '0; tag_tail_r <= '0; tag_count_r <= '0;
            issue_cursor_r <= '0;
            done <= 1'b0; error <= 1'b0;
            for (entry = 0; entry < ROB_DEPTH; entry = entry + 1) begin
                rob_valid[entry] <= 1'b0;
                rob_issued[entry] <= 1'b0;
                rob_ready[entry] <= 1'b0;
                rob_nibble[entry] <= 1'b0;
                rob_addr[entry] <= '0;
                rob_dest[entry] <= '0;
                rob_data[entry] <= '0;
            end
        end else begin
            done <= 1'b0;

            if (issue_fire) begin
                rob_issued[issue_index] <= 1'b1;
                tag_fifo[tag_tail_r] <= issue_index;
                tag_tail_r <= tag_tail_r + 1'b1;
                issue_cursor_r <= issue_cursor_r + 1'b1;
            end

            if (response_fire) begin
                if (!read_data_last) begin
                    error <= 1'b1;
                    state <= FAIL;
                end else begin
                    rob_data[tag_fifo[tag_head_r]] <= read_data;
                    rob_ready[tag_fifo[tag_head_r]] <= 1'b1;
                    tag_head_r <= tag_head_r + 1'b1;
                end
            end

            unique case ({issue_fire, response_fire})
                2'b10: tag_count_r <= tag_count_r + 1'b1;
                2'b01: tag_count_r <= tag_count_r - 1'b1;
                default: tag_count_r <= tag_count_r;
            endcase

            if (!issue_fire && issue_cursor_r < generated_count_r &&
                (rob_issued[issue_index] || rob_ready[issue_index]))
                issue_cursor_r <= issue_cursor_r + 1'b1;

            if (output_fire) begin
                rob_valid[emit_index] <= 1'b0;
                emitted_count_r <= emitted_count_r + 1'b1;
                if (plan_done_r && emitted_count_r + 1'b1 == total_count_r) begin
                    active_r <= 1'b0;
                    done <= 1'b1;
                    state <= IDLE;
                end
            end

            unique case (state)
                IDLE: begin
                    if (start) begin
                        error <= 1'b0;
                        active_r <= 1'b1;
                        plan_done_r <= 1'b0;
                        matrix_r <= matrix_select;
                        row_r <= '0; col_r <= '0;
                        rows_r <= tile_rows; cols_r <= tile_cols;
                        generated_count_r <= '0; emitted_count_r <= '0;
                        total_count_r <= tile_rows * tile_cols;
                        input_base_r <= input_base_addr;
                        weight_base_r <= weight_base_addr;
                        in_c_r <= input_c; in_h_r <= input_h;
                        in_w_r <= input_w; out_c_r <= out_c;
                        out_w_r <= out_w;
                        kh_size_r <= kernel_h; kw_size_r <= kernel_w;
                        sh_r <= stride_h; sw_r <= stride_w;
                        ph_r <= pad_h; pw_r <= pad_w;
                        n0_r <= n0;
                        kernel_area_r <= kernel_h * kernel_w;
                        m_remaining_r <= m0; k_remaining_r <= k0;
                        base_oh_r <= '0; base_ow_r <= '0;
                        base_ic_r <= '0; base_kh_r <= '0; base_kw_r <= '0;
                        tag_head_r <= '0; tag_tail_r <= '0;
                        tag_count_r <= '0; issue_cursor_r <= '0;
                        for (entry = 0; entry < ROB_DEPTH; entry = entry + 1) begin
                            rob_valid[entry] <= 1'b0;
                            rob_issued[entry] <= 1'b0;
                            rob_ready[entry] <= 1'b0;
                        end
                        if (tile_rows == 0 || tile_rows > TILE_SIZE ||
                            tile_cols == 0 || tile_cols > TILE_SIZE ||
                            input_c == 0 || input_h == 0 || input_w == 0 ||
                            out_c == 0 || out_w == 0 ||
                            kernel_h == 0 || kernel_w == 0)
                            state <= FAIL;
                        else
                            state <= DECODE_M;
                    end
                end

                DECODE_M: begin
                    if (m_remaining_r >= out_w_r) begin
                        m_remaining_r <= m_remaining_r - out_w_r;
                        base_oh_r <= base_oh_r + 1'b1;
                    end else begin
                        base_ow_r <= m_remaining_r;
                        row_oh_r <= base_oh_r;
                        row_ow_r <= m_remaining_r;
                        state <= DECODE_K_IC;
                    end
                end

                DECODE_K_IC: begin
                    if (k_remaining_r >= kernel_area_r) begin
                        k_remaining_r <= k_remaining_r - kernel_area_r;
                        base_ic_r <= base_ic_r + 1'b1;
                    end else begin
                        state <= DECODE_K_KH;
                    end
                end

                DECODE_K_KH: begin
                    if (k_remaining_r >= kw_size_r) begin
                        k_remaining_r <= k_remaining_r - kw_size_r;
                        base_kh_r <= base_kh_r + 1'b1;
                    end else begin
                        base_kw_r <= k_remaining_r;
                        k_ic_r <= base_ic_r;
                        k_kh_r <= base_kh_r;
                        k_kw_r <= k_remaining_r;
                        state <= PREPARE;
                    end
                end

                PREPARE: begin
                    pad_r <= 1'b0;
                    if (!matrix_r)
                        state <= COORD_MUL;
                    else if (k_ic_r >= in_c_r || n0_r + col_r >= out_c_r) begin
                        pad_r <= 1'b1;
                        byte_addr_r <= '0;
                        nibble_select_r <= 1'b0;
                        state <= RECORD;
                    end else
                        state <= ADDR_HORNER0;
                end

                COORD_MUL: begin
                    coord_h_product_r <= row_oh_r * sh_r;
                    coord_w_product_r <= row_ow_r * sw_r;
                    state <= COORD_ADD;
                end

                COORD_ADD: begin
                    input_y_r <= $signed({1'b0, coord_h_product_r[15:0]}) +
                                 $signed({1'b0, k_kh_r}) -
                                 $signed({1'b0, ph_r});
                    input_x_r <= $signed({1'b0, coord_w_product_r[15:0]}) +
                                 $signed({1'b0, k_kw_r}) -
                                 $signed({1'b0, pw_r});
                    state <= CHECK_BOUNDS;
                end

                CHECK_BOUNDS: begin
                    if (k_ic_r >= in_c_r ||
                        input_y_r < 0 ||
                        input_y_r >= $signed({1'b0, in_h_r}) ||
                        input_x_r < 0 ||
                        input_x_r >= $signed({1'b0, in_w_r})) begin
                        pad_r <= 1'b1;
                        byte_addr_r <= '0;
                        nibble_select_r <= 1'b0;
                        state <= RECORD;
                    end else
                        state <= ADDR_HORNER0;
                end

                // Registered Horner stages keep one multiply-add per cycle.
                // A: ((ic * H) + y) * W + x
                // B: (((oc * IC) + ic) * KH + kh) * KW + kw
                ADDR_HORNER0: begin
                    horner0_r <= matrix_r ?
                        ((n0_r + col_r) * in_c_r + k_ic_r) :
                        (k_ic_r * in_h_r + $unsigned(input_y_r));
                    state <= ADDR_HORNER1;
                end
                ADDR_HORNER1: begin
                    if (matrix_r) begin
                        horner1_r <=
                            horner0_r * kh_size_r + k_kh_r;
                        state <= ADDR_HORNER2;
                    end else begin
                        logical_index_r <=
                            horner0_r * in_w_r + $unsigned(input_x_r);
                        state <= PACK_ADDRESS;
                    end
                end
                ADDR_HORNER2: begin
                    logical_index_r <=
                        horner1_r * kw_size_r + k_kw_r;
                    state <= PACK_ADDRESS;
                end
                PACK_ADDRESS: begin
                    byte_addr_r <= (matrix_r ? weight_base_r : input_base_r) +
                                   (logical_index_r >> 1);
                    nibble_select_r <= logical_index_r[0];
                    state <= RECORD;
                end

                RECORD: begin
                    if (generated_count_r - emitted_count_r < ROB_DEPTH) begin
                        rob_valid[generated_count_r[ROB_PTR_W-1:0]] <= 1'b1;
                        rob_issued[generated_count_r[ROB_PTR_W-1:0]] <= pad_r;
                        rob_ready[generated_count_r[ROB_PTR_W-1:0]] <= pad_r;
                        rob_nibble[generated_count_r[ROB_PTR_W-1:0]] <= nibble_select_r;
                        rob_addr[generated_count_r[ROB_PTR_W-1:0]] <= byte_addr_r;
                        rob_dest[generated_count_r[ROB_PTR_W-1:0]] <=
                            row_r * TILE_SIZE + col_r;
                        rob_data[generated_count_r[ROB_PTR_W-1:0]] <= 8'd0;
                        generated_count_r <= generated_count_r + 1'b1;
                        state <= ADVANCE;
                    end
                end

                ADVANCE: begin
                    if (row_r == rows_r - 1'b1 &&
                        col_r == cols_r - 1'b1) begin
                        plan_done_r <= 1'b1;
                        state <= WAIT_DRAIN;
                    end else if (!matrix_r) begin
                        if (col_r == cols_r - 1'b1) begin
                            col_r <= '0;
                            row_r <= row_r + 1'b1;
                            increment_spatial();
                            k_ic_r <= base_ic_r;
                            k_kh_r <= base_kh_r;
                            k_kw_r <= base_kw_r;
                        end else begin
                            col_r <= col_r + 1'b1;
                            increment_k();
                        end
                        state <= PREPARE;
                    end else if (col_r == cols_r - 1'b1) begin
                        col_r <= '0;
                        row_r <= row_r + 1'b1;
                        increment_k();
                        state <= PREPARE;
                    end else begin
                        col_r <= col_r + 1'b1;
                        state <= PREPARE;
                    end
                end

                WAIT_DRAIN: begin
                    if (emitted_count_r == total_count_r) begin
                        active_r <= 1'b0;
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end

                FAIL: begin
                    error <= 1'b1;
                    active_r <= 1'b0;
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= FAIL;
            endcase
        end
    end
endmodule
