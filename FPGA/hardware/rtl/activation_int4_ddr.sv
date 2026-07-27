`timescale 1ns/1ps
module activation_int4_ddr (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        leaky_mode,
    input  logic [2:0]  leaky_shift,
    input  logic [31:0] input_base_addr,
    input  logic [31:0] output_base_addr,
    input  logic [31:0] element_count,
    output logic        busy,
    output logic        done,
    output logic        error,

    output logic        read_req_valid,
    input  logic        read_req_ready,
    output logic [31:0] read_req_addr,
    input  logic        read_data_valid,
    output logic        read_data_ready,
    input  logic [7:0]  read_data,

    output logic        write_req_valid,
    input  logic        write_req_ready,
    output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data,
    output logic [3:0]  write_req_strb
);
    typedef enum logic [2:0] {
        IDLE, READ_REQ, READ_WAIT, ACTIVATE, WRITE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [31:0] input_base_r, output_base_r, count_r, block_base_r;
    logic leaky_r;
    logic [2:0] shift_r;
    logic [2:0] byte_idx;
    logic [31:0] input_word_r, output_word_r;
    logic [3:0] valid_bytes_r;
    logic signed [31:0] activated_word;

    activation_int4_parallel #(.LANES(8)) activation (
        .enable(1'b1),
        .leaky_mode(leaky_r),
        .leaky_shift(shift_r),
        .input_flat(input_word_r),
        .output_flat(activated_word)
    );

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = input_base_r + (block_base_r >> 1) + byte_idx;
    assign read_data_ready = state == READ_WAIT;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = output_base_r + ((block_base_r >> 3) << 2);
    assign write_req_data = output_word_r;
    assign write_req_strb = valid_bytes_r;

    always_ff @(posedge clk) begin
        logic [31:0] remaining;
        logic [3:0] byte_count;
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            input_base_r <= '0;
            output_base_r <= '0;
            count_r <= '0;
            block_base_r <= '0;
            leaky_r <= 1'b0;
            shift_r <= '0;
            byte_idx <= '0;
            input_word_r <= '0;
            output_word_r <= '0;
            valid_bytes_r <= '0;
        end else begin
            done <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        input_base_r <= input_base_addr;
                        output_base_r <= output_base_addr;
                        count_r <= element_count;
                        block_base_r <= '0;
                        leaky_r <= leaky_mode;
                        shift_r <= leaky_shift;
                        byte_idx <= '0;
                        input_word_r <= '0;
                        if (element_count == 0 || output_base_addr[1:0] != 0) begin
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
                    remaining = count_r - block_base_r;
                    byte_count = remaining >= 8 ? 4 : ((remaining + 1) >> 1);
                    if (byte_idx + 1 >= byte_count) begin
                        valid_bytes_r <= (4'b0001 << byte_count) - 1'b1;
                        byte_idx <= '0;
                        state <= ACTIVATE;
                    end else begin
                        byte_idx <= byte_idx + 1'b1;
                        state <= READ_REQ;
                    end
                end

                ACTIVATE: begin
                    output_word_r <= activated_word;
                    state <= WRITE;
                end

                WRITE: if (write_req_ready) begin
                    if (block_base_r + 8 >= count_r) begin
                        state <= FINISH;
                    end else begin
                        block_base_r <= block_base_r + 8;
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
