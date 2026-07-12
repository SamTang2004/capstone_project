module tile_loader_ddr #(
    parameter int ADDR_WIDTH = 32,
    parameter int TILE_SIZE = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start,
    input  logic                    matrix_select,
    input  logic [ADDR_WIDTH-1:0]   base_addr,
    input  logic [ADDR_WIDTH-1:0]   row_stride,
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
    output logic [7:0]              cache_addr,
    output logic [7:0]              cache_wdata
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_REQ,
        STATE_WAIT_DATA,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;
    logic matrix_latched;
    logic [4:0] row_idx;
    logic [4:0] col_idx;
    logic [4:0] rows_latched;
    logic [4:0] cols_latched;
    logic [ADDR_WIDTH-1:0] base_latched;
    logic [ADDR_WIDTH-1:0] stride_latched;

    assign busy = (state == STATE_REQ || state == STATE_WAIT_DATA);
    assign read_req_valid = (state == STATE_REQ);
    assign read_req_addr = base_latched + (row_idx * stride_latched) + col_idx;
    assign read_data_ready = (state == STATE_WAIT_DATA);
    assign cache_we = read_data_valid && read_data_ready;
    assign cache_matrix = matrix_latched;
    assign cache_addr = (row_idx * TILE_SIZE) + col_idx;
    assign cache_wdata = read_data;

    function automatic logic is_last_element(
        input logic [4:0] row,
        input logic [4:0] col
    );
        is_last_element = (row == rows_latched - 5'd1) && (col == cols_latched - 5'd1);
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            matrix_latched <= 1'b0;
            row_idx <= '0;
            col_idx <= '0;
            rows_latched <= '0;
            cols_latched <= '0;
            base_latched <= '0;
            stride_latched <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    row_idx <= '0;
                    col_idx <= '0;
                    if (start) begin
                        matrix_latched <= matrix_select;
                        base_latched <= base_addr;
                        stride_latched <= row_stride;
                        rows_latched <= tile_rows;
                        cols_latched <= tile_cols;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 || tile_cols > TILE_SIZE || row_stride < tile_cols) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            state <= STATE_REQ;
                        end
                    end
                end

                STATE_REQ: begin
                    if (read_req_ready) begin
                        state <= STATE_WAIT_DATA;
                    end
                end

                STATE_WAIT_DATA: begin
                    if (read_data_valid) begin
                        if (is_last_element(row_idx, col_idx)) begin
                            state <= STATE_DONE;
                        end else if (col_idx == cols_latched - 5'd1) begin
                            col_idx <= '0;
                            row_idx <= row_idx + 5'd1;
                            state <= STATE_REQ;
                        end else begin
                            col_idx <= col_idx + 5'd1;
                            state <= STATE_REQ;
                        end
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
