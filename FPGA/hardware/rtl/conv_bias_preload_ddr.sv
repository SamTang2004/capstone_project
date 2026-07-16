// Seeds a 16x16 accumulator tile with INT32 Conv2D bias values from DDR.
module conv_bias_preload_ddr #(
    parameter int TILE_SIZE = 16
) (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] bias_base_addr, input logic [15:0] channel_offset,
    input logic [4:0] tile_rows, input logic [4:0] tile_cols,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    output logic accum_valid, input logic accum_ready, output logic [7:0] accum_addr, output logic signed [31:0] accum_data
);
    typedef enum logic [2:0] { IDLE, ADDR, READ_REQ, READ_WAIT, ACCUM, FINISH, FAIL } state_t;
    state_t state;
    logic [31:0] base_r, bias_word_r;
    logic [31:0] read_addr_r;
    logic [15:0] channel_r;
    logic [4:0] row_r, col_r, rows_r, cols_r;
    logic [1:0] byte_r;
    assign busy = state == READ_REQ || state == READ_WAIT || state == ACCUM;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = read_addr_r;
    assign read_data_ready = state == READ_WAIT || (state == READ_REQ && read_data_valid);
    assign accum_valid = state == ACCUM;
    assign accum_addr = row_r * TILE_SIZE + col_r;
    assign accum_data = bias_word_r;
    always_ff @(posedge clk) begin
        logic [31:0] assembled;
        if (!rst_n) begin
            state <= IDLE; base_r <= '0; bias_word_r <= '0; read_addr_r <= '0; channel_r <= '0; row_r <= '0; col_r <= '0; rows_r <= '0; cols_r <= '0; byte_r <= '0; done <= 1'b0; error <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    error <= 1'b0; row_r <= '0; col_r <= '0; byte_r <= '0;
                    if (start) begin
                        base_r <= bias_base_addr; channel_r <= channel_offset; rows_r <= tile_rows; cols_r <= tile_cols;
                        if (bias_base_addr[1:0] != 0 || tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 || tile_cols > TILE_SIZE) begin error <= 1'b1; state <= FAIL; end
                        else state <= ADDR;
                    end
                end
                ADDR: begin
                    read_addr_r <= base_r + (channel_r << 2) + byte_r;
                    state <= READ_REQ;
                end
                READ_REQ: begin
                    if (read_data_valid) state <= READ_REQ;
                    else if (read_req_ready) state <= READ_WAIT;
                end
                READ_WAIT: if (read_data_valid) begin
                    bias_word_r[byte_r*8 +: 8] <= read_data;
                    if (byte_r == 2'd3) begin
                        assembled = {read_data, bias_word_r[23:0]};
                        bias_word_r <= assembled; row_r <= '0; state <= ACCUM;
                    end else begin byte_r <= byte_r + 1'b1; state <= ADDR; end
                end
                ACCUM: if (accum_ready) begin
                    if (row_r == rows_r - 1'b1) begin
                        row_r <= '0;
                        if (col_r == cols_r - 1'b1) state <= FINISH;
                        else begin col_r <= col_r + 1'b1; channel_r <= channel_r + 1'b1; byte_r <= '0; state <= ADDR; end
                    end else row_r <= row_r + 1'b1;
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
