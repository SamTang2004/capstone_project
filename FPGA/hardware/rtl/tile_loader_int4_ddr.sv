`timescale 1ns/1ps
module tile_loader_int4_ddr #(
    parameter int ADDR_WIDTH = 32,
    parameter int TILE_SIZE = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic                    matrix_select,
    input  logic [ADDR_WIDTH-1:0]   base_addr,
    input  logic [ADDR_WIDTH-1:0]   row_stride_bytes,
    input  logic [4:0]              tile_rows,
    input  logic [4:0]              tile_cols,
    output logic                    busy,
    output logic                    done,
    output logic                    error,
    output logic                    read_req_valid,
    input  logic                    read_req_ready,
    output logic [ADDR_WIDTH-1:0]   read_req_addr,
    input  logic                    read_data_valid,
    output logic                    read_data_ready,
    input  logic [7:0]              read_data,
    output logic                    cache_we,
    output logic                    cache_matrix,
    output logic [5:0]              cache_addr,
    output logic signed [3:0]       cache_wdata
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_REQ,
        STATE_WAIT,
        STATE_EMIT,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;
    logic matrix_r;
    logic [3:0] row_r;
    logic [3:0] col_r;
    logic [3:0] rows_r;
    logic [3:0] cols_r;
    logic [ADDR_WIDTH-1:0] base_r;
    logic [ADDR_WIDTH-1:0] stride_r;
    logic [7:0] packed_byte_r;

    wire [3:0] byte_col = col_r >> 1;
    wire [ADDR_WIDTH-1:0] minimum_stride = (tile_cols + 1'b1) >> 1;

    assign busy = state == STATE_REQ || state == STATE_WAIT || state == STATE_EMIT;
    assign read_req_valid = state == STATE_REQ;
    assign read_req_addr = base_r + row_r * stride_r + byte_col;
    assign read_data_ready = state == STATE_WAIT;
    assign cache_we = state == STATE_EMIT;
    assign cache_matrix = matrix_r;
    assign cache_addr = row_r * TILE_SIZE + col_r;
    assign cache_wdata = col_r[0] ? $signed(packed_byte_r[7:4]) : $signed(packed_byte_r[3:0]);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            matrix_r <= 1'b0;
            row_r <= '0;
            col_r <= '0;
            rows_r <= '0;
            cols_r <= '0;
            base_r <= '0;
            stride_r <= '0;
            packed_byte_r <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    row_r <= '0;
                    col_r <= '0;
                    if (start) begin
                        matrix_r <= matrix_select;
                        rows_r <= tile_rows[3:0];
                        cols_r <= tile_cols[3:0];
                        base_r <= base_addr;
                        stride_r <= row_stride_bytes;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE ||
                            tile_cols == 0 || tile_cols > TILE_SIZE ||
                            row_stride_bytes < minimum_stride) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            state <= STATE_REQ;
                        end
                    end
                end

                STATE_REQ: if (read_req_ready) state <= STATE_WAIT;

                STATE_WAIT: if (read_data_valid) begin
                    packed_byte_r <= read_data;
                    state <= STATE_EMIT;
                end

                STATE_EMIT: begin
                    if (row_r == rows_r - 1'b1 && col_r == cols_r - 1'b1) begin
                        state <= STATE_DONE;
                    end else if (col_r == cols_r - 1'b1) begin
                        row_r <= row_r + 1'b1;
                        col_r <= '0;
                        state <= STATE_REQ;
                    end else begin
                        col_r <= col_r + 1'b1;
                        state <= col_r[0] ? STATE_REQ : STATE_EMIT;
                    end
                end

                STATE_DONE: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                STATE_ERROR: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
