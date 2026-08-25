module gemm_requantize_writer (
    input logic clk, rst_n, pipeline_enable, start,
    input logic [2047:0] accumulators, input logic [255:0] bias_values,
    input logic bias_enable, input logic [3:0] valid_rows, valid_cols,
    input logic [1:0] output_mode, input logic [31:0] output_row_stride,
    input logic [31:0] output_base_addr, input logic signed [31:0] multiplier,
    input logic [5:0] right_shift, input logic signed [31:0] output_zero_point,
    output logic busy, done, capture_done, error,
    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last
);
    typedef enum logic [2:0] {IDLE, FILL, DRAIN, PREP_WRITE, WRITE_TILE, FINISH, FAIL} state_t;
    state_t state;
    logic bias_enable_r;
    logic [1:0] mode_r;
    logic [3:0] rows_r, cols_r;
    logic [2:0] feed_row_r, feed_col_r;
    logic [31:0] base_r, row_stride_r;
    logic signed [31:0] multiplier_r, zero_point_r;
    logic [5:0] shift_r;

    logic v0, v1, v2, v3, vb0, vb1, vb2, vb3;
    logic last0, last1, last2, last3;
    logic [2:0] row0, row1, row2, row3, col0, col1, col2, col3;
    logic [2:0] rowb0, rowb1, rowb2, rowb3, colb0, colb1, colb2, colb3;
    logic signed [31:0] sum_r, multiply_lhs_r, multiply_rhs_r;
    logic signed [31:0] sum_b_r, multiply_lhs_b_r, multiply_rhs_b_r;
    (* use_dsp = "yes" *) logic signed [63:0] product_r;
    (* use_dsp = "yes" *) logic signed [63:0] product_b_r;
    logic signed [63:0] scaled_r;
    logic signed [63:0] scaled_b_r;

    // Store one row-major byte copy of the complete 8x8 output tile.  The
    // registered write-data port selects either rows or columns.  Keeping one
    // copy is important on the Zynq-7020: the former duplicate row/column word
    // banks cost 512 extra flip-flops and enough routing to hurt timing.
    logic [7:0] output_bytes [0:63];
    logic [4:0] write_index_r, write_count_r;
    logic [31:0] write_data_r;
    logic [3:0] write_strb_r;
    logic [2:0] write_row_r;
    logic [1:0] row_word_r, row_word_count_r;
    logic [31:0] row_byte_base_r;

    function automatic logic signed [7:0] saturate_int8(input logic signed [63:0] value);
        if (value < -64'sd128) saturate_int8 = -8'sd128;
        else if (value > 64'sd127) saturate_int8 = 8'sd127;
        else saturate_int8 = value[7:0];
    endfunction

    wire feeding = state == FILL;
    wire feed_lane_b = feeding && feed_row_r + 1'b1 < rows_r;
    wire feed_last = feed_row_r + 2 >= rows_r && feed_col_r + 1'b1 == cols_r;
    wire [31:0] next_row_byte_base = row_byte_base_r + row_stride_r;

    function automatic logic [31:0] aligned_row_word(
        input logic [2:0] row_index, input logic [1:0] word_index,
        input logic [1:0] byte_offset
    );
        logic [31:0] result;
        integer column, destination;
        begin
            result = '0;
            for (column = 0; column < 8; column = column + 1) begin
                destination = byte_offset + column;
                if (column < cols_r && (destination >> 2) == word_index)
                    result[(destination & 3)*8 +: 8] = output_bytes[row_index*8 + column];
            end
            aligned_row_word = result;
        end
    endfunction

    function automatic logic [31:0] packed_output_word(
        input logic [4:0] word_index, input logic transpose
    );
        logic [31:0] result;
        integer lane, row, column;
        begin
            result = '0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (transpose) begin
                    column = word_index >> 1;
                    row = ((word_index & 1) << 2) + lane;
                end else begin
                    row = word_index >> 1;
                    column = ((word_index & 1) << 2) + lane;
                end
                result[lane*8 +: 8] = output_bytes[row*8 + column];
            end
            packed_output_word = result;
        end
    endfunction

    function automatic logic [3:0] aligned_row_strb(
        input logic [1:0] word_index, input logic [1:0] byte_offset
    );
        logic [3:0] result;
        integer column, destination;
        begin
            result = '0;
            for (column = 0; column < 8; column = column + 1) begin
                destination = byte_offset + column;
                if (column < cols_r && (destination >> 2) == word_index)
                    result[destination & 3] = 1'b1;
            end
            aligned_row_strb = result;
        end
    endfunction

    // FINISH still owns the writer for one cycle while it publishes done.
    // Keeping busy asserted prevents a following GEMM tile from presenting a
    // start pulse that the FINISH state cannot accept.
    assign busy = state != IDLE && state != FAIL;
    assign write_req_valid = pipeline_enable && state == WRITE_TILE;
    assign write_req_addr = mode_r == 2'd1
        ? {row_byte_base_r[31:2], 2'b0} + (row_word_r << 2)
        : base_r + (write_index_r << 2);
    assign write_req_data = write_data_r;
    assign write_req_strb = write_strb_r;
    assign write_req_last = mode_r == 2'd1
        ? row_word_r + 1'b1 == row_word_count_r
        : write_index_r + 1'b1 == write_count_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 0; capture_done <= 0; error <= 0; bias_enable_r <= 0; mode_r <= 0;
            rows_r <= 0; cols_r <= 0; feed_row_r <= 0; feed_col_r <= 0; base_r <= 0; row_stride_r <= 0;
            multiplier_r <= 0; zero_point_r <= 0; shift_r <= 0;
            v0 <= 0; v1 <= 0; v2 <= 0; v3 <= 0;
            vb0 <= 0; vb1 <= 0; vb2 <= 0; vb3 <= 0;
            last0 <= 0; last1 <= 0; last2 <= 0; last3 <= 0;
            row0 <= 0; row1 <= 0; row2 <= 0; row3 <= 0;
            col0 <= 0; col1 <= 0; col2 <= 0; col3 <= 0;
            rowb0 <= 0; rowb1 <= 0; rowb2 <= 0; rowb3 <= 0;
            colb0 <= 0; colb1 <= 0; colb2 <= 0; colb3 <= 0;
            sum_r <= 0; multiply_lhs_r <= 0; multiply_rhs_r <= 0;
            sum_b_r <= 0; multiply_lhs_b_r <= 0; multiply_rhs_b_r <= 0;
            product_r <= 0; product_b_r <= 0; scaled_r <= 0; scaled_b_r <= 0;
            write_index_r <= 0; write_count_r <= 0; write_data_r <= 0; write_strb_r <= 0;
            write_row_r <= 0; row_word_r <= 0; row_word_count_r <= 0; row_byte_base_r <= 0;
            for (int byte_index = 0; byte_index < 64; byte_index++) output_bytes[byte_index] <= 0;
        end else begin
            done <= 0;
            capture_done <= 0;
            if (pipeline_enable) begin
                if (state == FILL || state == DRAIN) begin
                    // Explicit A/B/P registers remain inside the DSP chain, but
                    // every stage advances concurrently for one result per clock.
                    v0 <= feeding; v1 <= v0; v2 <= v1; v3 <= v2;
                    vb0 <= feed_lane_b; vb1 <= vb0; vb2 <= vb1; vb3 <= vb2;
                    last0 <= feeding && feed_last;
                    last1 <= last0; last2 <= last1; last3 <= last2;
                    row1 <= row0; row2 <= row1; row3 <= row2;
                    col1 <= col0; col2 <= col1; col3 <= col2;
                    rowb1 <= rowb0; rowb2 <= rowb1; rowb3 <= rowb2;
                    colb1 <= colb0; colb2 <= colb1; colb3 <= colb2;
                    if (feeding) begin
                        row0 <= feed_row_r; col0 <= feed_col_r;
                        rowb0 <= feed_row_r + 1'b1; colb0 <= feed_col_r;
                        sum_r <= accumulators[(feed_row_r*8+feed_col_r)*32 +: 32] +
                            (bias_enable_r ? bias_values[feed_col_r*32 +: 32] : 32'd0);
                        if (feed_lane_b)
                            sum_b_r <= accumulators[((feed_row_r+1'b1)*8+feed_col_r)*32 +: 32] +
                                (bias_enable_r ? bias_values[feed_col_r*32 +: 32] : 32'd0);
                        if (feed_last) state <= DRAIN;
                        else if (feed_row_r + 2 >= rows_r) begin
                            feed_row_r <= 0; feed_col_r <= feed_col_r + 1'b1;
                        end else feed_row_r <= feed_row_r + 2;
                    end
                    if (v0) begin multiply_lhs_r <= sum_r; multiply_rhs_r <= multiplier_r; end
                    if (vb0) begin multiply_lhs_b_r <= sum_b_r; multiply_rhs_b_r <= multiplier_r; end
                    if (v1) product_r <= multiply_lhs_r * multiply_rhs_r;
                    if (vb1) product_b_r <= multiply_lhs_b_r * multiply_rhs_b_r;
                    if (v2) scaled_r <= (product_r >>> shift_r) + zero_point_r;
                    if (vb2) scaled_b_r <= (product_b_r >>> shift_r) + zero_point_r;
                    if (v3) begin
                        output_bytes[row3*8 + col3] <= saturate_int8(scaled_r);
                        if (vb3)
                            output_bytes[rowb3*8 + colb3] <= saturate_int8(scaled_b_r);
                        if (last3) begin
                            capture_done <= 1'b1;
                            state <= PREP_WRITE;
                        end
                    end
                end

                unique case (state)
                    IDLE: if (start) begin
                        error <= 0;
                        if (valid_rows == 0 || valid_cols == 0 || right_shift > 31 ||
                            multiplier[31] || output_base_addr[1:0] != 0) begin
                            error <= 1; state <= FAIL;
                        end else begin
                            bias_enable_r <= bias_enable; rows_r <= valid_rows; cols_r <= valid_cols;
                            mode_r <= output_mode; base_r <= output_base_addr; row_stride_r <= output_row_stride;
                            multiplier_r <= multiplier;
                            shift_r <= right_shift; zero_point_r <= output_zero_point;
                            feed_row_r <= 0; feed_col_r <= 0;
                            v0 <= 0; v1 <= 0; v2 <= 0; v3 <= 0;
                            vb0 <= 0; vb1 <= 0; vb2 <= 0; vb3 <= 0;
                            last0 <= 0; last1 <= 0; last2 <= 0; last3 <= 0;
                            write_index_r <= 0;
                            for (int byte_index = 0; byte_index < 64; byte_index++) output_bytes[byte_index] <= 0;
                            state <= FILL;
                        end
                    end
                    PREP_WRITE: begin
                        write_index_r <= 0; write_row_r <= 0; row_word_r <= 0;
                        if (mode_r == 2'd1) begin
                            row_byte_base_r <= base_r;
                            row_word_count_r <= (base_r[1:0] + cols_r + 3) >> 2;
                            write_data_r <= aligned_row_word(0, 0, base_r[1:0]);
                            write_strb_r <= aligned_row_strb(0, base_r[1:0]);
                        end else begin
                            write_count_r <= mode_r == 2'd2 ? {rows_r, 1'b0} : {cols_r, 1'b0};
                            write_data_r <= packed_output_word(0, mode_r != 2'd2);
                            write_strb_r <= 4'hf;
                        end
                        state <= WRITE_TILE;
                    end
                    WRITE_TILE: if (write_req_valid && write_req_ready) begin
                        if (mode_r == 2'd1) begin
                            if (row_word_r + 1'b1 == row_word_count_r) begin
                                if (write_row_r + 1'b1 == rows_r) state <= FINISH;
                                else begin
                                    write_row_r <= write_row_r + 1'b1; row_word_r <= 0;
                                    row_byte_base_r <= next_row_byte_base;
                                    row_word_count_r <= (next_row_byte_base[1:0] + cols_r + 3) >> 2;
                                    write_data_r <= aligned_row_word(write_row_r + 1'b1, 0,
                                        next_row_byte_base[1:0]);
                                    write_strb_r <= aligned_row_strb(0, next_row_byte_base[1:0]);
                                end
                            end else begin
                                row_word_r <= row_word_r + 1'b1;
                                write_data_r <= aligned_row_word(write_row_r, row_word_r + 1'b1, row_byte_base_r[1:0]);
                                write_strb_r <= aligned_row_strb(row_word_r + 1'b1, row_byte_base_r[1:0]);
                            end
                        end else if (write_index_r + 1'b1 == write_count_r) state <= FINISH;
                        else begin
                            write_index_r <= write_index_r + 1'b1;
                            write_data_r <= packed_output_word(write_index_r + 1'b1, mode_r != 2'd2);
                        end
                    end
                    FINISH: begin done <= 1; state <= IDLE; end
                    FAIL: begin done <= 1; if (!start) state <= IDLE; end
                    default: begin end
                endcase
            end
        end
    end
endmodule
