module tile_writer_ddr #(
    parameter int ADDR_WIDTH = 32,
    parameter int TILE_SIZE = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start,
    input  logic [ADDR_WIDTH-1:0]   base_addr,
    input  logic [ADDR_WIDTH-1:0]   row_stride,
    input  logic [4:0]              tile_rows,
    input  logic [4:0]              tile_cols,
    output logic                    busy,
    output logic                    done,
    output logic                    error,

    output logic                    tile_read_req,
    output logic [7:0]              tile_read_addr,
    input  logic                    tile_read_valid,
    input  logic signed [31:0]      tile_read_data,

    output logic                    write_req_valid,
    output logic                    write_req_last,
    input  logic                    write_req_ready,
    output logic [ADDR_WIDTH-1:0]   write_req_addr,
    output logic signed [31:0]      write_req_data
);

    typedef enum logic [2:0] { STATE_IDLE, STATE_READ_REQ, STATE_READ_WAIT, STATE_WRITE, STATE_DONE, STATE_ERROR } state_t;
    state_t state;
    logic [4:0] row_idx, col_idx, rows_latched, cols_latched;
    logic [ADDR_WIDTH-1:0] base_latched, stride_latched;
    logic [ADDR_WIDTH-1:0] write_addr_latched;
    logic signed [31:0] write_data_latched;

    assign busy = (state == STATE_READ_REQ || state == STATE_READ_WAIT || state == STATE_WRITE);
    assign tile_read_req = (state == STATE_READ_REQ);
    assign tile_read_addr = (row_idx * TILE_SIZE) + col_idx;
    assign write_req_valid = (state == STATE_WRITE);
    assign write_req_last = (state == STATE_WRITE) && is_last_element(row_idx, col_idx);
    assign write_req_addr = write_addr_latched;
    assign write_req_data = write_data_latched;

    function automatic logic is_last_element(input logic [4:0] row, input logic [4:0] col);
        is_last_element = (row == rows_latched - 5'd1) && (col == cols_latched - 5'd1);
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            row_idx <= '0; col_idx <= '0; rows_latched <= '0; cols_latched <= '0;
            base_latched <= '0; stride_latched <= '0;
            write_addr_latched <= '0; write_data_latched <= '0;
            done <= 1'b0; error <= 1'b0;
        end else begin
            done <= 1'b0;
            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0; row_idx <= '0; col_idx <= '0;
                    if (start) begin
                        base_latched <= base_addr; stride_latched <= row_stride;
                        rows_latched <= tile_rows; cols_latched <= tile_cols;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 || tile_cols > TILE_SIZE || row_stride < tile_cols * 32'd4) begin
                            error <= 1'b1; state <= STATE_ERROR;
                        end else state <= STATE_READ_REQ;
                    end
                end
                STATE_READ_REQ: state <= STATE_READ_WAIT;
                STATE_READ_WAIT: if (tile_read_valid) begin
                    write_addr_latched <= base_latched + (row_idx * stride_latched) + (col_idx * 32'd4);
                    write_data_latched <= tile_read_data;
                    state <= STATE_WRITE;
                end
                STATE_WRITE: if (write_req_ready) begin
                    if (is_last_element(row_idx, col_idx)) state <= STATE_DONE;
                    else if (col_idx == cols_latched - 5'd1) begin col_idx <= '0; row_idx <= row_idx + 5'd1; state <= STATE_READ_REQ; end
                    else begin col_idx <= col_idx + 5'd1; state <= STATE_READ_REQ; end
                end
                STATE_DONE: begin done <= 1'b1; state <= STATE_IDLE; end
                STATE_ERROR: begin done <= 1'b1; state <= STATE_IDLE; end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
