// Restores NCHW layout after the systolic core produces a spatial-row x channel tile.
// Spatial decode and output address formation are sequential and fully registered.
module conv2d_tile_writer_ddr #(
    parameter int TILE_SIZE = 16
) (
    input logic clk, input logic rst_n, input logic start, input logic conv_mode,
    input logic [31:0] base_addr, input logic [31:0] row_stride,
    input logic [31:0] output_base_addr, input logic [15:0] out_h, input logic [15:0] out_w,
    input logic [15:0] m0, input logic [15:0] n0, input logic [4:0] tile_rows, input logic [4:0] tile_cols,
    output logic busy, output logic done, output logic error,
    output logic tile_read_req, output logic [7:0] tile_read_addr,
    input logic tile_read_valid, input logic signed [31:0] tile_read_data,
    output logic write_req_valid, output logic write_req_last, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic signed [31:0] write_req_data
);
    typedef enum logic [4:0] {
        IDLE, DECODE_M, READ_REQ, READ_WAIT, ADDR_PREP,
        ADDR_MUL0_PREP, ADDR_MUL0_0, ADDR_MUL0_1, ADDR_MUL0_ADD01,
        ADDR_MUL0_2, ADDR_MUL0_3, ADDR_MUL0_ADD23, ADDR_MUL0_COMBINE, ADDR_ADD0,
        ADDR_MUL1_PREP, ADDR_MUL1_0, ADDR_MUL1_1, ADDR_MUL1_ADD01,
        ADDR_MUL1_2, ADDR_MUL1_3, ADDR_MUL1_ADD23, ADDR_MUL1_COMBINE,
        ADDR_ADD1, ADDR_BASE,
        WRITE, FINISH, FAIL
    } state_t;
    state_t state;
    logic [4:0] row_r, col_r, rows_r, cols_r;
    logic [31:0] base_r, stride_r, write_addr_r;
    logic conv_r;
    logic [15:0] out_h_r, out_w_r, n0_r;
    logic [15:0] m_remaining_r, base_oh_r, base_ow_r, row_oh_r, row_ow_r;
    logic [31:0] addr_index_r, addr_mul0_r, addr_term0_r, addr_mul1_r, addr_term1_r;
    logic [31:0] addr_mul_lhs_r;
    logic [15:0] addr_mul_rhs_r;
    logic [23:0] addr_mul_chunk0_r, addr_mul_chunk1_r;
    logic [31:0] addr_mul_pair_low_r, addr_mul_pair_high_r;
    logic signed [31:0] write_data_r;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign tile_read_req = state == READ_REQ;
    assign tile_read_addr = row_r * TILE_SIZE + col_r;
    assign write_req_valid = state == WRITE;
    // Each NCHW output channel is one contiguous burst run.
    assign write_req_last = state == WRITE && (conv_r ? row_r == rows_r - 1'b1 :
                                               (row_r == rows_r - 1'b1 && col_r == cols_r - 1'b1));
    assign write_req_addr = write_addr_r;
    assign write_req_data = write_data_r;

    task automatic increment_spatial;
        begin
            if (row_ow_r + 1 >= out_w_r) begin row_ow_r <= '0; row_oh_r <= row_oh_r + 1'b1; end
            else row_ow_r <= row_ow_r + 1'b1;
        end
    endtask

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; row_r <= '0; col_r <= '0; rows_r <= '0; cols_r <= '0;
            base_r <= '0; stride_r <= '0; write_addr_r <= '0; conv_r <= 1'b0;
            out_h_r <= '0; out_w_r <= '0; n0_r <= '0; m_remaining_r <= '0;
            base_oh_r <= '0; base_ow_r <= '0; row_oh_r <= '0; row_ow_r <= '0;
            addr_index_r <= '0; addr_mul0_r <= '0; addr_term0_r <= '0;
            addr_mul1_r <= '0; addr_term1_r <= '0; write_data_r <= '0;
            addr_mul_lhs_r <= '0; addr_mul_rhs_r <= '0; addr_mul_chunk0_r <= '0; addr_mul_chunk1_r <= '0;
            addr_mul_pair_low_r <= '0; addr_mul_pair_high_r <= '0;
            done <= 1'b0; error <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    error <= 1'b0; row_r <= '0; col_r <= '0; base_oh_r <= '0; base_ow_r <= '0;
                    if (start) begin
                        base_r <= conv_mode ? output_base_addr : base_addr;
                        stride_r <= row_stride; conv_r <= conv_mode;
                        out_h_r <= out_h; out_w_r <= out_w; n0_r <= n0;
                        m_remaining_r <= m0; rows_r <= tile_rows; cols_r <= tile_cols;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 || tile_cols > TILE_SIZE ||
                            (conv_mode && (out_h == 0 || out_w == 0))) begin error <= 1'b1; state <= FAIL; end
                        else state <= conv_mode ? DECODE_M : READ_REQ;
                    end
                end
                DECODE_M: begin
                    if (m_remaining_r >= out_w_r) begin
                        m_remaining_r <= m_remaining_r - out_w_r;
                        base_oh_r <= base_oh_r + 1'b1;
                    end else begin
                        base_ow_r <= m_remaining_r; row_oh_r <= base_oh_r; row_ow_r <= m_remaining_r;
                        state <= READ_REQ;
                    end
                end
                // Accumulator storage is synchronous block RAM. Issue the
                // address, then wait for its registered response.
                READ_REQ: state <= READ_WAIT;
                READ_WAIT: if (tile_read_valid) begin write_data_r <= tile_read_data; state <= ADDR_PREP; end
                // Fully registered Horner pipeline. Multiplies and additions
                // occupy separate cycles so no tile index drives a combined
                // multiply-add cone.
                ADDR_PREP: begin
                    addr_index_r <= conv_r ? n0_r + col_r : row_r;
                    state <= ADDR_MUL0_PREP;
                end
                ADDR_MUL0_PREP: begin
                    addr_mul_lhs_r <= conv_r ? addr_index_r : stride_r;
                    addr_mul_rhs_r <= conv_r ? out_h_r : addr_index_r[15:0];
                    state <= ADDR_MUL0_0;
                end
                ADDR_MUL0_0: begin addr_mul_chunk0_r <= addr_mul_lhs_r[7:0] * addr_mul_rhs_r; state <= ADDR_MUL0_1; end
                ADDR_MUL0_1: begin addr_mul_chunk1_r <= addr_mul_lhs_r[15:8] * addr_mul_rhs_r; state <= ADDR_MUL0_ADD01; end
                ADDR_MUL0_ADD01: begin addr_mul_pair_low_r <= {8'd0, addr_mul_chunk0_r} + ({8'd0, addr_mul_chunk1_r} << 8); state <= ADDR_MUL0_2; end
                ADDR_MUL0_2: begin addr_mul_chunk0_r <= addr_mul_lhs_r[23:16] * addr_mul_rhs_r; state <= ADDR_MUL0_3; end
                ADDR_MUL0_3: begin addr_mul_chunk1_r <= addr_mul_lhs_r[31:24] * addr_mul_rhs_r; state <= ADDR_MUL0_ADD23; end
                ADDR_MUL0_ADD23: begin addr_mul_pair_high_r <= {8'd0, addr_mul_chunk0_r} + ({8'd0, addr_mul_chunk1_r} << 8); state <= ADDR_MUL0_COMBINE; end
                ADDR_MUL0_COMBINE: begin addr_mul0_r <= addr_mul_pair_low_r + (addr_mul_pair_high_r << 16); state <= ADDR_ADD0; end
                ADDR_ADD0: begin
                    addr_term0_r <= conv_r ? addr_mul0_r + row_oh_r : addr_mul0_r;
                    state <= conv_r ? ADDR_MUL1_PREP : ADDR_ADD1;
                end
                ADDR_MUL1_PREP: begin addr_mul_lhs_r <= addr_term0_r; addr_mul_rhs_r <= out_w_r; state <= ADDR_MUL1_0; end
                ADDR_MUL1_0: begin addr_mul_chunk0_r <= addr_mul_lhs_r[7:0] * addr_mul_rhs_r; state <= ADDR_MUL1_1; end
                ADDR_MUL1_1: begin addr_mul_chunk1_r <= addr_mul_lhs_r[15:8] * addr_mul_rhs_r; state <= ADDR_MUL1_ADD01; end
                ADDR_MUL1_ADD01: begin addr_mul_pair_low_r <= {8'd0, addr_mul_chunk0_r} + ({8'd0, addr_mul_chunk1_r} << 8); state <= ADDR_MUL1_2; end
                ADDR_MUL1_2: begin addr_mul_chunk0_r <= addr_mul_lhs_r[23:16] * addr_mul_rhs_r; state <= ADDR_MUL1_3; end
                ADDR_MUL1_3: begin addr_mul_chunk1_r <= addr_mul_lhs_r[31:24] * addr_mul_rhs_r; state <= ADDR_MUL1_ADD23; end
                ADDR_MUL1_ADD23: begin addr_mul_pair_high_r <= {8'd0, addr_mul_chunk0_r} + ({8'd0, addr_mul_chunk1_r} << 8); state <= ADDR_MUL1_COMBINE; end
                ADDR_MUL1_COMBINE: begin addr_mul1_r <= addr_mul_pair_low_r + (addr_mul_pair_high_r << 16); state <= ADDR_ADD1; end
                ADDR_ADD1: begin
                    addr_term1_r <= conv_r ? addr_mul1_r + row_ow_r : addr_term0_r + (col_r << 2);
                    state <= ADDR_BASE;
                end
                ADDR_BASE: begin
                    write_addr_r <= base_r + (conv_r ? (addr_term1_r << 2) : addr_term1_r);
                    state <= WRITE;
                end
                WRITE: if (write_req_ready) begin
                    if (row_r == rows_r - 1'b1 && col_r == cols_r - 1'b1) state <= FINISH;
                    else if (conv_r && row_r == rows_r - 1'b1) begin
                        row_r <= '0; col_r <= col_r + 1'b1;
                        row_oh_r <= base_oh_r; row_ow_r <= base_ow_r; state <= READ_REQ;
                    end else if (!conv_r && col_r == cols_r - 1'b1) begin
                        col_r <= '0; row_r <= row_r + 1'b1; state <= READ_REQ;
                    end else if (conv_r) begin
                        row_r <= row_r + 1'b1; increment_spatial(); state <= READ_REQ;
                    end else begin col_r <= col_r + 1'b1; state <= READ_REQ; end
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
