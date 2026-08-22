// Reads up to sixteen INT8 values in four-byte beats, applies activation in
// parallel, then emits packed writes. Requests never cross a 16-byte block.
module activation_ddr_parallel #(
    parameter int LANES = 16
) (
    input logic clk, input logic rst_n, input logic start,
    input logic leaky_mode, input logic [2:0] leaky_shift,
    input logic [31:0] input_base_addr, input logic [31:0] output_base_addr,
    input logic [31:0] element_count,
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
    typedef enum logic [2:0] {
        IDLE, READ_REQ, READ_RECEIVE, WRITE_PREP, WRITE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [31:0] in_base_r, out_base_r, count_r, block_base_r;
    logic [4:0] block_len_r, receive_remaining_r, receive_lane_r;
    logic leaky_r;
    logic [2:0] shift_r;
    logic [1:0] word_r;
    logic signed [LANES*8-1:0] input_flat, output_flat;
    logic [31:0] write_addr_r, write_data_r;
    logic [3:0] write_strb_r;
    logic last_word_r, last_block_r;

    wire [2:0] response_bytes = read_word_keep[0] + read_word_keep[1] +
                                read_word_keep[2] + read_word_keep[3];
    wire response_shape_valid = response_bytes != 0 &&
        read_word_keep == ((5'd1 << response_bytes) - 1'b1);

    activation_unfold_parallel #(.LANES(LANES)) activation (
        .enable(1'b1), .leaky_mode(leaky_r), .leaky_shift(shift_r),
        .input_flat(input_flat), .output_flat(output_flat)
    );

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = in_base_r + block_base_r;
    assign read_req_len = block_len_r;
    assign read_word_ready = state == READ_RECEIVE;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = write_data_r;
    assign write_req_strb = write_strb_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            in_base_r <= '0;
            out_base_r <= '0;
            count_r <= '0;
            block_base_r <= '0;
            block_len_r <= '0;
            receive_remaining_r <= '0;
            receive_lane_r <= '0;
            leaky_r <= 1'b0;
            shift_r <= '0;
            word_r <= '0;
            input_flat <= '0;
            write_addr_r <= '0;
            write_data_r <= '0;
            write_strb_r <= '0;
            last_word_r <= 1'b0;
            last_block_r <= 1'b0;
        end else begin
            done <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        in_base_r <= input_base_addr;
                        out_base_r <= output_base_addr;
                        count_r <= element_count;
                        block_base_r <= '0;
                        block_len_r <= element_count >= LANES ? LANES : element_count[4:0];
                        receive_lane_r <= '0;
                        input_flat <= '0;
                        leaky_r <= leaky_mode;
                        shift_r <= leaky_shift;
                        if (element_count == 0 || LANES != 16) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else begin
                            state <= READ_REQ;
                        end
                    end
                end

                READ_REQ: if (read_req_ready) begin
                    receive_remaining_r <= block_len_r;
                    receive_lane_r <= '0;
                    state <= READ_RECEIVE;
                end

                READ_RECEIVE: if (read_word_valid) begin
                    if (!response_shape_valid || response_bytes > receive_remaining_r ||
                        read_word_last != (response_bytes == receive_remaining_r)) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        for (int lane = 0; lane < 4; lane++) begin
                            if (read_word_keep[lane])
                                input_flat[(receive_lane_r + lane)*8 +: 8] <=
                                    read_word_data[lane*8 +: 8];
                        end
                        receive_lane_r <= receive_lane_r + response_bytes;
                        receive_remaining_r <= receive_remaining_r - response_bytes;
                        if (response_bytes == receive_remaining_r) begin
                            word_r <= '0;
                            state <= WRITE_PREP;
                        end
                    end
                end

                WRITE_PREP: begin
                    write_addr_r <= out_base_r + block_base_r + (word_r << 2);
                    write_data_r <= output_flat[word_r*32 +: 32];
                    write_strb_r <= 4'hf;
                    for (int lane = 0; lane < 4; lane++) begin
                        if ((word_r << 2) + lane >= block_len_r)
                            write_strb_r[lane] <= 1'b0;
                    end
                    last_word_r <= (word_r << 2) + 4 >= block_len_r;
                    last_block_r <= block_base_r + block_len_r >= count_r;
                    state <= WRITE;
                end

                WRITE: if (write_req_ready) begin
                    if (last_word_r) begin
                        if (last_block_r) begin
                            state <= FINISH;
                        end else begin
                            block_base_r <= block_base_r + block_len_r;
                            block_len_r <= count_r - (block_base_r + block_len_r) >= LANES
                                ? LANES : count_r - (block_base_r + block_len_r);
                            input_flat <= '0;
                            state <= READ_REQ;
                        end
                    end else begin
                        word_r <= word_r + 1'b1;
                        state <= WRITE_PREP;
                    end
                end

                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
