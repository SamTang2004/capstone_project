`timescale 1ns/1ps
module requantize_int4_ddr (
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
    typedef enum logic [3:0] {
        IDLE, READ_REQ, READ_WAIT, QUANT_LAUNCH, QUANT_WAIT,
        PACK, WRITE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [31:0] input_base_r, output_base_r, count_r, element_idx;
    logic signed [31:0] multiplier_r, zero_point_r;
    logic [5:0] shift_r;
    logic [1:0] byte_idx;
    logic [31:0] input_word_r, packed_word_r;
    logic quant_input_valid, quant_output_valid;
    logic signed [3:0] quant_output;
    logic [31:0] write_addr_r;
    logic [3:0] write_strb_r;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = input_base_r + (element_idx << 2) + byte_idx;
    assign read_data_ready = state == READ_WAIT;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = packed_word_r;
    assign write_req_strb = write_strb_r;

    requantize_int4_pipeline quantizer (
        .clk, .rst_n,
        .input_valid(quant_input_valid),
        .input_value($signed(input_word_r)),
        .multiplier(multiplier_r),
        .right_shift(shift_r),
        .output_zero_point(zero_point_r),
        .output_valid(quant_output_valid),
        .output_value(quant_output)
    );

    always_ff @(posedge clk) begin
        logic [31:0] next_word;
        logic [3:0] byte_count;
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            input_base_r <= '0;
            output_base_r <= '0;
            count_r <= '0;
            element_idx <= '0;
            multiplier_r <= '0;
            zero_point_r <= '0;
            shift_r <= '0;
            byte_idx <= '0;
            input_word_r <= '0;
            packed_word_r <= '0;
            quant_input_valid <= 1'b0;
            write_addr_r <= '0;
            write_strb_r <= '0;
        end else begin
            done <= 1'b0;
            quant_input_valid <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        input_base_r <= input_base_addr;
                        output_base_r <= output_base_addr;
                        count_r <= element_count;
                        multiplier_r <= multiplier;
                        zero_point_r <= output_zero_point;
                        shift_r <= right_shift;
                        element_idx <= '0;
                        byte_idx <= '0;
                        input_word_r <= '0;
                        packed_word_r <= '0;
                        if (element_count == 0 || right_shift > 31 ||
                            input_base_addr[1:0] != 0 ||
                            output_base_addr[1:0] != 0) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else begin
                            state <= READ_REQ;
                        end
                    end
                end

                READ_REQ: if (read_req_ready) state <= READ_WAIT;

                READ_WAIT: if (read_data_valid) begin
                    input_word_r[byte_idx*8 +: 8] <= read_data;
                    if (byte_idx == 2'd3) begin
                        byte_idx <= '0;
                        state <= QUANT_LAUNCH;
                    end else begin
                        byte_idx <= byte_idx + 1'b1;
                        state <= READ_REQ;
                    end
                end

                QUANT_LAUNCH: begin
                    quant_input_valid <= 1'b1;
                    state <= QUANT_WAIT;
                end

                QUANT_WAIT: if (quant_output_valid) state <= PACK;

                PACK: begin
                    next_word = packed_word_r;
                    next_word[element_idx[2:0]*4 +: 4] = quant_output;
                    packed_word_r <= next_word;
                    if (element_idx[2:0] == 3'd7 ||
                        element_idx + 1 >= count_r) begin
                        byte_count = (element_idx[2:0] + 2) >> 1;
                        write_addr_r <= output_base_r +
                            ((element_idx >> 3) << 2);
                        write_strb_r <= element_idx[2:0] == 3'd7
                            ? 4'hf : ((4'b0001 << byte_count) - 1'b1);
                        packed_word_r <= next_word;
                        state <= WRITE;
                    end else begin
                        element_idx <= element_idx + 1'b1;
                        input_word_r <= '0;
                        state <= READ_REQ;
                    end
                end

                WRITE: if (write_req_ready) begin
                    packed_word_r <= '0;
                    if (element_idx + 1 >= count_r) begin
                        state <= FINISH;
                    end else begin
                        element_idx <= element_idx + 1'b1;
                        input_word_r <= '0;
                        state <= READ_REQ;
                    end
                end

                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
