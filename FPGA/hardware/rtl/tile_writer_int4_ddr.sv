`timescale 1ns/1ps
module tile_writer_int4_ddr #(
    parameter int ADDR_WIDTH = 32,
    parameter int TILE_SIZE = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic [ADDR_WIDTH-1:0]   base_addr,
    input  logic [ADDR_WIDTH-1:0]   row_stride_bytes,
    input  logic [4:0]              tile_rows,
    input  logic [4:0]              tile_cols,
    output logic                    busy,
    output logic                    done,
    output logic                    error,
    output logic                    tile_read_req,
    output logic [5:0]              tile_read_addr,
    input  logic                    tile_read_valid,
    input  logic signed [3:0]       tile_read_data,
    output logic                    write_req_valid,
    output logic                    write_req_last,
    input  logic                    write_req_ready,
    output logic [ADDR_WIDTH-1:0]   write_req_addr,
    output logic [7:0]              write_req_data
);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_READ_LOW_REQ,
        STATE_READ_LOW_WAIT,
        STATE_READ_HIGH_REQ,
        STATE_READ_HIGH_WAIT,
        STATE_WRITE,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;
    logic [3:0] row_r;
    logic [2:0] byte_col_r;
    logic [3:0] rows_r;
    logic [3:0] cols_r;
    logic [3:0] bytes_per_row_r;
    logic [ADDR_WIDTH-1:0] base_r;
    logic [ADDR_WIDTH-1:0] stride_r;
    logic [3:0] low_nibble_r;
    logic [3:0] high_nibble_r;

    wire [3:0] low_col = {byte_col_r, 1'b0};
    wire [ADDR_WIDTH-1:0] minimum_stride = (tile_cols + 1'b1) >> 1;

    assign busy = state != STATE_IDLE && state != STATE_DONE && state != STATE_ERROR;
    assign tile_read_req = state == STATE_READ_LOW_REQ || state == STATE_READ_HIGH_REQ;
    assign tile_read_addr = row_r * TILE_SIZE +
                            (state == STATE_READ_HIGH_REQ ? low_col + 1'b1 : low_col);
    assign write_req_valid = state == STATE_WRITE;
    assign write_req_last = state == STATE_WRITE &&
                            row_r == rows_r - 1'b1 &&
                            byte_col_r == bytes_per_row_r - 1'b1;
    assign write_req_addr = base_r + row_r * stride_r + byte_col_r;
    assign write_req_data = {high_nibble_r, low_nibble_r};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            row_r <= '0;
            byte_col_r <= '0;
            rows_r <= '0;
            cols_r <= '0;
            bytes_per_row_r <= '0;
            base_r <= '0;
            stride_r <= '0;
            low_nibble_r <= '0;
            high_nibble_r <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    row_r <= '0;
                    byte_col_r <= '0;
                    if (start) begin
                        rows_r <= tile_rows[3:0];
                        cols_r <= tile_cols[3:0];
                        bytes_per_row_r <= (tile_cols + 1'b1) >> 1;
                        base_r <= base_addr;
                        stride_r <= row_stride_bytes;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE ||
                            tile_cols == 0 || tile_cols > TILE_SIZE ||
                            row_stride_bytes < minimum_stride) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            state <= STATE_READ_LOW_REQ;
                        end
                    end
                end

                STATE_READ_LOW_REQ: state <= STATE_READ_LOW_WAIT;

                STATE_READ_LOW_WAIT: if (tile_read_valid) begin
                    low_nibble_r <= tile_read_data;
                    if (low_col + 1'b1 < cols_r)
                        state <= STATE_READ_HIGH_REQ;
                    else begin
                        high_nibble_r <= '0;
                        state <= STATE_WRITE;
                    end
                end

                STATE_READ_HIGH_REQ: state <= STATE_READ_HIGH_WAIT;

                STATE_READ_HIGH_WAIT: if (tile_read_valid) begin
                    high_nibble_r <= tile_read_data;
                    state <= STATE_WRITE;
                end

                STATE_WRITE: if (write_req_ready) begin
                    if (row_r == rows_r - 1'b1 &&
                        byte_col_r == bytes_per_row_r - 1'b1) begin
                        state <= STATE_DONE;
                    end else if (byte_col_r == bytes_per_row_r - 1'b1) begin
                        row_r <= row_r + 1'b1;
                        byte_col_r <= '0;
                        state <= STATE_READ_LOW_REQ;
                    end else begin
                        byte_col_r <= byte_col_r + 1'b1;
                        state <= STATE_READ_LOW_REQ;
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
