// Reads sixteen INT8 values, applies activation in parallel, then emits four packed writes.
module activation_ddr_parallel #(
    parameter int LANES = 16
) (
    input logic clk, input logic rst_n, input logic start, input logic leaky_mode, input logic [2:0] leaky_shift,
    input logic [31:0] input_base_addr, input logic [31:0] output_base_addr, input logic [31:0] element_count,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    output logic write_req_valid, input logic write_req_ready, output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data, output logic [3:0] write_req_strb
);
    typedef enum logic [2:0] { IDLE, READ_CHECK, READ_REQ, READ_WAIT, WRITE_CHECK, WRITE, FINISH, FAIL } state_t;
    state_t state;
    logic [31:0] in_base_r, out_base_r, count_r, block_base_r;
    logic leaky_r; logic [2:0] shift_r; logic [4:0] lane_r; logic [1:0] word_r;
    logic signed [LANES*8-1:0] input_flat, output_flat;
    logic [31:0] read_addr_r, write_addr_r, write_data_r;
    logic [3:0] write_strb_r;
    logic lane_valid_r, last_word_r, last_block_r;
    activation_unfold_parallel #(.LANES(LANES)) activation (.enable(1'b1), .leaky_mode(leaky_r), .leaky_shift(shift_r), .input_flat(input_flat), .output_flat(output_flat));
    assign busy = state == READ_REQ || state == READ_WAIT || state == WRITE;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = read_addr_r;
    assign read_data_ready = state == READ_WAIT || (state == READ_REQ && read_data_valid);
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = write_data_r;
    assign write_req_strb = write_strb_r;
    always_ff @(posedge clk) begin
        if (!rst_n) begin state <= IDLE; done <= 1'b0; error <= 1'b0; in_base_r <= '0; out_base_r <= '0; count_r <= '0; block_base_r <= '0; leaky_r <= 1'b0; shift_r <= '0; lane_r <= '0; word_r <= '0; input_flat <= '0; read_addr_r <= '0; write_addr_r <= '0; write_data_r <= '0; write_strb_r <= '0; lane_valid_r <= 1'b0; last_word_r <= 1'b0; last_block_r <= 1'b0; end
        else begin
            done <= 1'b0;
            case (state)
                IDLE: begin error <= 1'b0; if (start) begin in_base_r <= input_base_addr; out_base_r <= output_base_addr; count_r <= element_count; block_base_r <= '0; lane_r <= '0; word_r <= '0; leaky_r <= leaky_mode; shift_r <= leaky_shift; if (element_count == 0) begin error <= 1'b1; state <= FAIL; end else state <= READ_CHECK; end end
                READ_CHECK: begin
                    read_addr_r <= in_base_r + block_base_r + lane_r;
                    lane_valid_r <= block_base_r + lane_r < count_r;
                    state <= READ_REQ;
                end
                READ_REQ: if (read_data_valid) state <= READ_REQ; else if (!lane_valid_r) begin input_flat[lane_r*8 +: 8] <= '0; if (lane_r == LANES-1) begin word_r <= '0; state <= WRITE_CHECK; end else begin lane_r <= lane_r + 1'b1; state <= READ_CHECK; end end else if (read_req_ready) state <= READ_WAIT;
                READ_WAIT: if (read_data_valid) begin input_flat[lane_r*8 +: 8] <= read_data; if (lane_r == LANES-1) begin word_r <= '0; state <= WRITE_CHECK; end else begin lane_r <= lane_r + 1'b1; state <= READ_CHECK; end end
                WRITE_CHECK: begin
                    write_addr_r <= out_base_r + block_base_r + (word_r << 2);
                    write_data_r <= output_flat[word_r*32 +: 32];
                    write_strb_r <= 4'hf;
                    for (int b = 0; b < 4; b++)
                        if (block_base_r + (word_r << 2) + b >= count_r) write_strb_r[b] <= 1'b0;
                    last_word_r <= word_r == 2'd3 || block_base_r + (word_r << 2) + 4 >= count_r;
                    last_block_r <= block_base_r + LANES >= count_r;
                    state <= WRITE;
                end
                WRITE: if (write_req_ready) begin if (last_word_r) begin if (last_block_r) state <= FINISH; else begin block_base_r <= block_base_r + LANES; lane_r <= '0; state <= READ_CHECK; end end else begin word_r <= word_r + 1'b1; state <= WRITE_CHECK; end end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
