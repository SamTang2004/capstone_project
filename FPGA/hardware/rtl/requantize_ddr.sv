module requantize_ddr (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] input_base_addr, input logic [31:0] output_base_addr,
    input logic [31:0] element_count,
    input logic signed [31:0] multiplier,
    input logic [5:0] right_shift,
    input logic signed [31:0] output_zero_point,
    output logic busy, output logic done, output logic error,

    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [4:0] read_req_len,
    input logic read_word_valid, output logic read_word_ready,
    input logic [31:0] read_word_data, input logic [3:0] read_word_keep,
    input logic read_word_last,

    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb
);
    typedef enum logic [3:0] {
        IDLE, REQUEST, RECEIVE, LOAD_INPUT, MULTIPLY, SCALE, PACK,
        WRITE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [31:0] input_base_r, output_base_r, count_r, element_base_r;
    logic signed [31:0] multiplier_r, output_zero_point_r;
    logic [5:0] shift_r;
    logic [2:0] batch_elements_r, batch_index_r;
    logic [4:0] request_len_r, receive_remaining_r, receive_offset_r;
    logic [127:0] input_batch_r;
    logic signed [31:0] input_value_r;
    (* use_dsp = "yes" *) logic signed [63:0] product_r;
    logic signed [63:0] scaled_r;
    logic [31:0] packed_word_r;
    logic [3:0] write_strb_r;

    wire [2:0] response_bytes = read_word_keep[0] + read_word_keep[1] +
                                read_word_keep[2] + read_word_keep[3];
    wire response_shape_valid = response_bytes != 0 &&
        read_word_keep == ((5'd1 << response_bytes) - 1'b1);

    function automatic logic signed [7:0] saturate_int8(
        input logic signed [63:0] scaled
    );
        if (scaled < -64'sd128) saturate_int8 = -8'sd128;
        else if (scaled > 64'sd127) saturate_int8 = 8'sd127;
        else saturate_int8 = scaled[7:0];
    endfunction

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == REQUEST;
    assign read_req_addr = input_base_r + (element_base_r << 2);
    assign read_req_len = request_len_r;
    assign read_word_ready = state == RECEIVE;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = output_base_r + element_base_r;
    assign write_req_data = packed_word_r;
    assign write_req_strb = write_strb_r;

    always_ff @(posedge clk) begin
        logic signed [7:0] quantized;
        logic [31:0] next_packed;
        logic [31:0] remaining_elements;
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            input_base_r <= '0;
            output_base_r <= '0;
            count_r <= '0;
            element_base_r <= '0;
            multiplier_r <= '0;
            output_zero_point_r <= '0;
            shift_r <= '0;
            batch_elements_r <= '0;
            batch_index_r <= '0;
            request_len_r <= '0;
            receive_remaining_r <= '0;
            receive_offset_r <= '0;
            input_batch_r <= '0;
            input_value_r <= '0;
            product_r <= '0;
            scaled_r <= '0;
            packed_word_r <= '0;
            write_strb_r <= '0;
        end else begin
            done <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        input_base_r <= input_base_addr;
                        output_base_r <= output_base_addr;
                        count_r <= element_count;
                        element_base_r <= '0;
                        multiplier_r <= multiplier;
                        output_zero_point_r <= output_zero_point;
                        shift_r <= right_shift;
                        batch_elements_r <= element_count >= 4 ? 3'd4 : element_count[2:0];
                        request_len_r <= element_count >= 4 ? 5'd16 : element_count[2:0] << 2;
                        packed_word_r <= '0;
                        input_batch_r <= '0;
                        if (element_count == 0 || right_shift > 31 ||
                            input_base_addr[1:0] != 0 || output_base_addr[1:0] != 0) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else begin
                            state <= REQUEST;
                        end
                    end
                end

                REQUEST: if (read_req_ready) begin
                    receive_remaining_r <= request_len_r;
                    receive_offset_r <= '0;
                    input_batch_r <= '0;
                    state <= RECEIVE;
                end

                RECEIVE: if (read_word_valid) begin
                    if (!response_shape_valid || response_bytes > receive_remaining_r ||
                        read_word_last != (response_bytes == receive_remaining_r)) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        for (int lane = 0; lane < 4; lane++) begin
                            if (read_word_keep[lane])
                                input_batch_r[(receive_offset_r + lane)*8 +: 8] <=
                                    read_word_data[lane*8 +: 8];
                        end
                        receive_offset_r <= receive_offset_r + response_bytes;
                        receive_remaining_r <= receive_remaining_r - response_bytes;
                        if (response_bytes == receive_remaining_r) begin
                            batch_index_r <= '0;
                            state <= LOAD_INPUT;
                        end
                    end
                end

                LOAD_INPUT: begin
                    input_value_r <= input_batch_r[batch_index_r*32 +: 32];
                    state <= MULTIPLY;
                end

                MULTIPLY: begin
                    product_r <= input_value_r * multiplier_r;
                    state <= SCALE;
                end

                SCALE: begin
                    scaled_r <= (product_r >>> shift_r) + output_zero_point_r;
                    state <= PACK;
                end

                PACK: begin
                    quantized = saturate_int8(scaled_r);
                    next_packed = packed_word_r;
                    next_packed[batch_index_r*8 +: 8] = quantized;
                    packed_word_r <= next_packed;
                    if (batch_index_r + 1 >= batch_elements_r) begin
                        write_strb_r <= (5'd1 << batch_elements_r) - 1'b1;
                        state <= WRITE;
                    end else begin
                        batch_index_r <= batch_index_r + 1'b1;
                        state <= LOAD_INPUT;
                    end
                end

                WRITE: if (write_req_ready) begin
                    if (element_base_r + batch_elements_r >= count_r) begin
                        state <= FINISH;
                    end else begin
                        remaining_elements = count_r - (element_base_r + batch_elements_r);
                        element_base_r <= element_base_r + batch_elements_r;
                        batch_elements_r <= remaining_elements >= 4 ? 3'd4 : remaining_elements[2:0];
                        request_len_r <= remaining_elements >= 4 ? 5'd16 : remaining_elements[2:0] << 2;
                        packed_word_r <= '0;
                        state <= REQUEST;
                    end
                end

                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
