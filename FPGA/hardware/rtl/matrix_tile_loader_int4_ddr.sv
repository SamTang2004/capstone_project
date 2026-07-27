`timescale 1ns/1ps
module matrix_tile_loader_int4_ddr #(
    parameter int ADDR_WIDTH = 32,
    parameter int TILE_SIZE = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic                    load_array,
    input  logic                    load_matrix,
    input  logic [ADDR_WIDTH-1:0]   tensor_base,
    input  logic [ADDR_WIDTH-1:0]   row_stride_bytes,
    input  logic [15:0]             row_offset,
    input  logic [15:0]             col_offset,
    input  logic [3:0]              tile_rows,
    input  logic [3:0]              tile_cols,
    output logic                    busy,
    output logic                    done,
    output logic                    error,

    output logic                    read_req_valid,
    input  logic                    read_req_ready,
    output logic [ADDR_WIDTH-1:0]   read_req_addr,
    output logic [4:0]              read_req_len,
    input  logic                    read_word_valid,
    output logic                    read_word_ready,
    input  logic [31:0]             read_word_data,
    input  logic [3:0]              read_word_keep,
    input  logic                    read_word_last,

    output logic                    vector_valid,
    input  logic                    vector_ready,
    output logic                    vector_array,
    output logic                    vector_matrix,
    output logic [2:0]              vector_row,
    output logic [7:0]              vector_mask,
    output logic [31:0]             vector_data
);
    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_ROW_ADD,
        STATE_ROW_MUL,
        STATE_ADDR,
        STATE_CHUNK,
        STATE_REQ,
        STATE_RECV,
        STATE_FAST_STREAM,
        STATE_EMIT,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;
    logic array_r;
    logic matrix_r;
    logic [2:0] row_r;
    logic [3:0] rows_r;
    logic [3:0] cols_r;
    logic [ADDR_WIDTH-1:0] base_r;
    logic [ADDR_WIDTH-1:0] stride_r;
    logic [15:0] row_offset_r;
    logic [15:0] col_offset_r;
    logic [ADDR_WIDTH-1:0] logical_row_r;
    logic [ADDR_WIDTH-1:0] row_product_r;
    logic [ADDR_WIDTH-1:0] request_addr_r;
    logic [4:0] remaining_bytes_r;
    logic [4:0] chunk_bytes_r;
    logic [3:0] gathered_bytes_r;
    logic [39:0] row_bytes_r;
    logic [3:0] requested_rows_r;
    logic [3:0] received_rows_r;

    function automatic logic [2:0] keep_count(input logic [3:0] keep);
        keep_count = keep[0] + keep[1] + keep[2] + keep[3];
    endfunction

    function automatic logic [7:0] active_mask(input logic [3:0] columns);
        if (columns >= 8)
            active_mask = 8'hff;
        else
            active_mask = (8'h01 << columns) - 1'b1;
    endfunction

    function automatic logic [ADDR_WIDTH-1:0] row_span(
        input logic [3:0] rows,
        input logic [ADDR_WIDTH-1:0] stride
    );
        case (rows)
            4'd1: row_span = '0;
            4'd2: row_span = stride;
            4'd3: row_span = (stride << 1);
            4'd4: row_span = (stride << 1) + stride;
            4'd5: row_span = (stride << 2);
            4'd6: row_span = (stride << 2) + stride;
            4'd7: row_span = (stride << 2) + (stride << 1);
            default: row_span = (stride << 3) - stride;
        endcase
    endfunction

    wire [2:0] accepted_bytes = keep_count(read_word_keep);
    wire [ADDR_WIDTH-1:0] calculated_row_addr =
        base_r + row_product_r + (col_offset_r >> 1);
    wire fast_rows_share_line =
        ({1'b0, calculated_row_addr[5:0]} +
         row_span(rows_r, stride_r) + 4) <= 64;

    assign busy = state != STATE_IDLE && state != STATE_DONE &&
                  state != STATE_ERROR;
    assign read_req_valid = state == STATE_REQ ||
                            (state == STATE_FAST_STREAM &&
                             requested_rows_r < rows_r);
    assign read_req_addr = request_addr_r;
    assign read_req_len = state == STATE_FAST_STREAM ? 5'd4 : chunk_bytes_r;
    assign read_word_ready = state == STATE_FAST_STREAM ?
                             vector_ready : state == STATE_RECV;
    assign vector_valid = state == STATE_EMIT ||
                          (state == STATE_FAST_STREAM && read_word_valid);
    assign vector_array = array_r;
    assign vector_matrix = matrix_r;
    assign vector_row = state == STATE_FAST_STREAM ?
                        received_rows_r[2:0] : row_r;
    assign vector_mask = state == STATE_FAST_STREAM ?
                         8'hff : active_mask(cols_r);
    assign vector_data = state == STATE_FAST_STREAM ?
                         read_word_data :
                         (col_offset_r[0] ?
                          row_bytes_r[35:4] : row_bytes_r[31:0]);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            array_r <= 1'b0;
            matrix_r <= 1'b0;
            row_r <= '0;
            rows_r <= '0;
            cols_r <= '0;
            base_r <= '0;
            stride_r <= '0;
            row_offset_r <= '0;
            col_offset_r <= '0;
            logical_row_r <= '0;
            row_product_r <= '0;
            request_addr_r <= '0;
            remaining_bytes_r <= '0;
            chunk_bytes_r <= '0;
            gathered_bytes_r <= '0;
            row_bytes_r <= '0;
            requested_rows_r <= '0;
            received_rows_r <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    row_r <= '0;
                    if (start) begin
                        array_r <= load_array;
                        matrix_r <= load_matrix;
                        rows_r <= tile_rows;
                        cols_r <= tile_cols;
                        base_r <= tensor_base;
                        stride_r <= row_stride_bytes;
                        row_offset_r <= row_offset;
                        col_offset_r <= col_offset;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE ||
                            tile_cols == 0 || tile_cols > TILE_SIZE ||
                            row_stride_bytes == 0) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            state <= STATE_ROW_ADD;
                        end
                    end
                end

                STATE_ROW_ADD: begin
                    logical_row_r <= row_offset_r + row_r;
                    row_bytes_r <= '0;
                    gathered_bytes_r <= '0;
                    state <= STATE_ROW_MUL;
                end

                STATE_ROW_MUL: begin
                    row_product_r <= logical_row_r * stride_r;
                    state <= STATE_ADDR;
                end

                STATE_ADDR: begin
                    request_addr_r <= calculated_row_addr;
                    if (cols_r == 8 && !col_offset_r[0] &&
                        fast_rows_share_line) begin
                        requested_rows_r <= {1'b0, row_r};
                        received_rows_r <= {1'b0, row_r};
                        state <= STATE_FAST_STREAM;
                    end else begin
                        remaining_bytes_r <=
                            ({1'b0, cols_r} +
                             {4'd0, col_offset_r[0]} + 1'b1) >> 1;
                        state <= STATE_CHUNK;
                    end
                end

                STATE_CHUNK: begin
                    if ({1'b0, request_addr_r[5:0]} + remaining_bytes_r > 64)
                        chunk_bytes_r <= 64 - request_addr_r[5:0];
                    else
                        chunk_bytes_r <= remaining_bytes_r;
                    state <= STATE_REQ;
                end

                STATE_REQ: if (read_req_ready)
                    state <= STATE_RECV;

                STATE_RECV: if (read_word_valid) begin
                    for (int lane = 0; lane < 4; lane++) begin
                        if (read_word_keep[lane])
                            row_bytes_r[(gathered_bytes_r + lane)*8 +: 8] <=
                                read_word_data[lane*8 +: 8];
                    end
                    gathered_bytes_r <= gathered_bytes_r + accepted_bytes;
                    if (read_word_last) begin
                        if (accepted_bytes == 0) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else if (remaining_bytes_r > chunk_bytes_r) begin
                            request_addr_r <= request_addr_r + chunk_bytes_r;
                            remaining_bytes_r <= remaining_bytes_r - chunk_bytes_r;
                            state <= STATE_CHUNK;
                        end else begin
                            state <= STATE_EMIT;
                        end
                    end
                end

                STATE_FAST_STREAM: begin
                    if (read_req_valid && read_req_ready) begin
                        requested_rows_r <= requested_rows_r + 1'b1;
                        request_addr_r <= request_addr_r + stride_r;
                    end
                    if (read_word_valid && vector_ready) begin
                        if (!read_word_last || read_word_keep != 4'hf) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else if (received_rows_r + 1'b1 >= rows_r) begin
                            state <= STATE_DONE;
                        end else begin
                            received_rows_r <= received_rows_r + 1'b1;
                        end
                    end
                end

                STATE_EMIT: if (vector_ready) begin
                    if (row_r + 1'b1 >= rows_r) begin
                        state <= STATE_DONE;
                    end else begin
                        row_r <= row_r + 1'b1;
                        state <= STATE_ROW_ADD;
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
