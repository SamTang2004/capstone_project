module requantize_pack_int8 (
    input  logic clk, rst_n, pipeline_enable, start,
    input  logic [31:0] input_base_addr, output_base_addr, input_row_stride,
    input  logic [15:0] rows, columns,
    input  logic signed [31:0] multiplier,
    input  logic [5:0] right_shift,
    input  logic signed [31:0] output_zero_point,
    output logic busy, done, error,
    output logic op_start, op_done, memory_stall, input_stall, output_stall,

    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [6:0] read_req_len,
    output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input logic read_word_valid, output logic read_word_ready,
    input logic [31:0] read_word_data, input logic [3:0] read_word_keep,
    input logic read_word_last,

    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last,
    input logic memory_busy, memory_error
);
    typedef enum logic [3:0] {
        IDLE, READ_REQ, READ_RECV, MULTIPLY, SCALE, STORE,
        WRITE_LOW, WRITE_HIGH, WRITE_ARM, WRITE_WAIT, FINISH, FAIL
    } state_t;

    state_t state;
    logic [15:0] row_index, column_index;
    logic [3:0] lane_index, valid_rows;
    logic [31:0] input_tile_addr, column_base_addr, lane_addr;
    logic [31:0] output_addr, input_tile_stride;
    logic [31:0] input_stride_r;
    logic [15:0] rows_r, columns_r;
    logic signed [31:0] multiplier_r, output_zero_point_r, input_value_r;
    logic [5:0] shift_r;
    (* use_dsp = "yes" *) logic signed [63:0] product_r;
    logic signed [63:0] scaled_r;
    logic [63:0] packed_vector_r;

    function automatic logic signed [7:0] saturate_int8(input logic signed [63:0] value);
        if (value < -64'sd128) saturate_int8 = -8'sd128;
        else if (value > 64'sd127) saturate_int8 = 8'sd127;
        else saturate_int8 = value[7:0];
    endfunction

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;
    assign read_req_valid = pipeline_enable && state == READ_REQ;
    assign read_req_addr = lane_addr;
    assign read_req_len = 7'd4;
    assign read_req_wide = 1'b1;
    assign read_req_qword = 1'b0;
    assign read_req_allow_ahead = 1'b0;
    assign read_word_ready = pipeline_enable && state == READ_RECV;
    assign write_req_valid = pipeline_enable && (state == WRITE_LOW || state == WRITE_HIGH);
    assign write_req_addr = output_addr;
    assign write_req_data = state == WRITE_LOW ? packed_vector_r[31:0] : packed_vector_r[63:32];
    assign write_req_strb = 4'hf;
    assign write_req_last = state == WRITE_HIGH;
    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
                                   (read_word_ready && !read_word_valid) ||
                                   (write_req_valid && !write_req_ready));
    assign input_stall = state == READ_RECV && !read_word_valid;
    assign output_stall = (state == WRITE_LOW || state == WRITE_HIGH) && !write_req_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            row_index <= '0; column_index <= '0; lane_index <= '0; valid_rows <= '0;
            input_tile_addr <= '0; column_base_addr <= '0; lane_addr <= '0;
            output_addr <= '0; input_tile_stride <= '0; input_stride_r <= '0;
            rows_r <= '0; columns_r <= '0; multiplier_r <= '0;
            output_zero_point_r <= '0; shift_r <= '0; input_value_r <= '0;
            product_r <= '0; scaled_r <= '0; packed_vector_r <= '0;
        end else begin
            done <= 1'b0;
            if (memory_error) begin error <= 1'b1; state <= FAIL; end
            else if (pipeline_enable) unique case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (rows == 0 || columns == 0 || right_shift > 31 ||
                        input_base_addr[1:0] != 0 || output_base_addr[1:0] != 0 ||
                        input_row_stride[1:0] != 0 || input_row_stride < (columns << 2)) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        rows_r <= rows; columns_r <= columns; input_stride_r <= input_row_stride;
                        multiplier_r <= multiplier; shift_r <= right_shift;
                        output_zero_point_r <= output_zero_point;
                        row_index <= 0; column_index <= 0; lane_index <= 0;
                        valid_rows <= rows >= 8 ? 8 : rows[3:0];
                        input_tile_addr <= input_base_addr;
                        column_base_addr <= input_base_addr;
                        lane_addr <= input_base_addr;
                        input_tile_stride <= input_row_stride << 3;
                        output_addr <= output_base_addr;
                        packed_vector_r <= '0;
                        state <= READ_REQ;
                    end
                end
                READ_REQ: if (read_req_valid && read_req_ready) state <= READ_RECV;
                READ_RECV: if (read_word_valid) begin
                    if (read_word_keep != 4'hf || !read_word_last) begin error <= 1'b1; state <= FAIL; end
                    else begin input_value_r <= read_word_data; state <= MULTIPLY; end
                end
                MULTIPLY: begin product_r <= input_value_r * multiplier_r; state <= SCALE; end
                SCALE: begin scaled_r <= (product_r >>> shift_r) + output_zero_point_r; state <= STORE; end
                STORE: begin
                    packed_vector_r[lane_index*8 +: 8] <= saturate_int8(scaled_r);
                    if (lane_index + 1'b1 == valid_rows) begin lane_index <= 0; state <= WRITE_LOW; end
                    else begin
                        lane_index <= lane_index + 1'b1;
                        lane_addr <= lane_addr + input_stride_r;
                        state <= READ_REQ;
                    end
                end
                WRITE_LOW: if (write_req_valid && write_req_ready) begin
                    output_addr <= output_addr + 4;
                    state <= WRITE_HIGH;
                end
                WRITE_HIGH: if (write_req_valid && write_req_ready) begin
                    output_addr <= output_addr + 4;
                    packed_vector_r <= '0;
                    if (column_index + 1'b1 == columns_r) begin
                        column_index <= 0;
                        if (row_index + valid_rows >= rows_r) state <= WRITE_ARM;
                        else begin
                            row_index <= row_index + 8;
                            input_tile_addr <= input_tile_addr + input_tile_stride;
                            column_base_addr <= input_tile_addr + input_tile_stride;
                            lane_addr <= input_tile_addr + input_tile_stride;
                            valid_rows <= rows_r - (row_index + 8) >= 8 ? 8 : rows_r - (row_index + 8);
                            state <= READ_REQ;
                        end
                    end else begin
                        column_index <= column_index + 1'b1;
                        column_base_addr <= column_base_addr + 4;
                        lane_addr <= column_base_addr + 4;
                        state <= READ_REQ;
                    end
                end
                WRITE_ARM: state <= WRITE_WAIT;
                WRITE_WAIT: if (!memory_busy) state <= FINISH;
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; if (!start) state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
