module requantize_ddr (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [31:0]        input_base_addr,
    input  logic [31:0]        output_base_addr,
    input  logic [31:0]        element_count,
    input  logic signed [31:0] multiplier,
    input  logic [5:0]         right_shift,
    input  logic signed [31:0] output_zero_point,
    output logic               busy,
    output logic               done,
    output logic               error,

    output logic               read_req_valid,
    input  logic               read_req_ready,
    output logic [31:0]        read_req_addr,
    input  logic               read_data_valid,
    output logic               read_data_ready,
    input  logic [7:0]         read_data,

    output logic               write_req_valid,
    input  logic               write_req_ready,
    output logic [31:0]        write_req_addr,
    output logic [31:0]        write_req_data,
    output logic [3:0]         write_req_strb
);

    typedef enum logic [3:0] { IDLE, REQUEST, RECEIVE, MULTIPLY, SCALE, PACK, WRITE, FINISH, FAIL } state_t;
    state_t state;
    logic [31:0] input_base_r, output_base_r, count_r;
    logic signed [31:0] multiplier_r, output_zero_point_r;
    logic [5:0] shift_r;
    logic [31:0] element_idx;
    logic [1:0] byte_idx;
    logic [31:0] input_word;
    logic signed [31:0] input_value_r;
    (* use_dsp = "yes" *) logic signed [63:0] product_r;
    logic signed [63:0] scaled_r;
    logic [31:0] packed_word;
    logic [31:0] write_addr_r;
    logic [3:0] write_strb_r;

    function automatic logic signed [7:0] saturate_int8(input logic signed [63:0] scaled);
        begin
            if (scaled < -64'sd128) begin
                saturate_int8 = -8'sd128;
            end else if (scaled > 64'sd127) begin
                saturate_int8 = 8'sd127;
            end else begin
                saturate_int8 = scaled[7:0];
            end
        end
    endfunction

    assign busy = state == REQUEST || state == RECEIVE || state == MULTIPLY || state == SCALE || state == PACK || state == WRITE;
    assign read_req_valid = state == REQUEST;
    assign read_req_addr = input_base_r + element_idx * 32'd4 + byte_idx;
    assign read_data_ready = state == RECEIVE;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = packed_word;
    assign write_req_strb = write_strb_r;

    always_ff @(posedge clk) begin
        logic signed [31:0] complete_input;
        logic signed [7:0] quantized;
        logic [31:0] next_packed;
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            input_base_r <= '0;
            output_base_r <= '0;
            count_r <= '0;
            multiplier_r <= '0;
            output_zero_point_r <= '0;
            shift_r <= '0;
            element_idx <= '0;
            byte_idx <= '0;
            input_word <= '0;
            input_value_r <= '0;
            product_r <= '0;
            scaled_r <= '0;
            packed_word <= '0;
            write_addr_r <= '0;
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
                        multiplier_r <= multiplier;
                        output_zero_point_r <= output_zero_point;
                        shift_r <= right_shift;
                        element_idx <= '0;
                        byte_idx <= '0;
                        packed_word <= '0;
                        if (element_count == 0 || right_shift > 31 || input_base_addr[1:0] != 0 || output_base_addr[1:0] != 0) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else begin
                            state <= REQUEST;
                        end
                    end
                end

                REQUEST: if (read_req_ready) state <= RECEIVE;

                RECEIVE: if (read_data_valid) begin
                    input_word[byte_idx * 8 +: 8] <= read_data;
                    if (byte_idx == 2'd3) begin
                        complete_input = {read_data, input_word[23:0]};
                        input_value_r <= complete_input;
                        byte_idx <= '0;
                        state <= MULTIPLY;
                    end else begin
                        byte_idx <= byte_idx + 1'b1;
                        state <= REQUEST;
                    end
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
                    next_packed = packed_word;
                    next_packed[element_idx[1:0] * 8 +: 8] = quantized;
                    packed_word <= next_packed;
                    if (element_idx[1:0] == 2'd3 || element_idx == count_r - 1) begin
                        write_addr_r <= output_base_r + (element_idx & 32'hffff_fffc);
                        write_strb_r <= element_idx[1:0] == 2'd3 ? 4'hf : ((4'b0001 << (element_idx[1:0] + 1'b1)) - 1'b1);
                        state <= WRITE;
                    end else begin
                        element_idx <= element_idx + 1'b1;
                        state <= REQUEST;
                    end
                end

                WRITE: if (write_req_ready) begin
                    packed_word <= '0;
                    if (element_idx == count_r - 1) begin
                        state <= FINISH;
                    end else begin
                        element_idx <= element_idx + 1'b1;
                        state <= REQUEST;
                    end
                end

                FINISH: begin
                    done <= 1'b1;
                    state <= IDLE;
                end

                FAIL: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
