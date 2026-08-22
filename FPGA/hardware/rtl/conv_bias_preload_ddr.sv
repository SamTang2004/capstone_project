// Seeds a 16x16 accumulator tile with INT32 Conv2D bias values from DDR.
module conv_bias_preload_ddr #(
    parameter int TILE_SIZE = 16
) (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] bias_base_addr, input logic [15:0] channel_offset,
    input logic [4:0] tile_rows, input logic [4:0] tile_cols,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    output logic [4:0] read_req_len,
    input logic read_word_valid, output logic read_word_ready,
    input logic [31:0] read_word_data, input logic [3:0] read_word_keep,
    input logic read_word_last,
    output logic accum_valid, input logic accum_ready, output logic [7:0] accum_addr, output logic signed [31:0] accum_data
);
    typedef enum logic [2:0] { IDLE, ADDR, READ_REQ, READ_WAIT, ACCUM, FINISH, FAIL } state_t;
    state_t state;
    logic [31:0] base_r, bias_word_r;
    logic [31:0] read_addr_r;
    logic [15:0] channel_r;
    logic [4:0] row_r, col_r, rows_r, cols_r;
    assign busy = state == READ_REQ || state == READ_WAIT || state == ACCUM;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = read_addr_r;
    assign read_req_len = 5'd4;
    assign read_word_ready = state == READ_WAIT;
    assign accum_valid = state == ACCUM;
    assign accum_addr = row_r * TILE_SIZE + col_r;
    assign accum_data = bias_word_r;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; base_r <= '0; bias_word_r <= '0; read_addr_r <= '0; channel_r <= '0; row_r <= '0; col_r <= '0; rows_r <= '0; cols_r <= '0; done <= 1'b0; error <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    error <= 1'b0; row_r <= '0; col_r <= '0;
                    if (start) begin
                        base_r <= bias_base_addr; channel_r <= channel_offset; rows_r <= tile_rows; cols_r <= tile_cols;
                        if (bias_base_addr[1:0] != 0 || tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 || tile_cols > TILE_SIZE) begin error <= 1'b1; state <= FAIL; end
                        else state <= ADDR;
                    end
                end
                ADDR: begin
                    read_addr_r <= base_r + (channel_r << 2);
                    state <= READ_REQ;
                end
                READ_REQ: if (read_req_ready) state <= READ_WAIT;
                READ_WAIT: if (read_word_valid) begin
                    if (read_word_keep != 4'hf || !read_word_last) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        bias_word_r <= read_word_data;
                        row_r <= '0;
                        state <= ACCUM;
                    end
                end
                ACCUM: if (accum_ready) begin
                    if (row_r == rows_r - 1'b1) begin
                        row_r <= '0;
                        if (col_r == cols_r - 1'b1) state <= FINISH;
                        else begin col_r <= col_r + 1'b1; channel_r <= channel_r + 1'b1; state <= ADDR; end
                    end else row_r <= row_r + 1'b1;
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
