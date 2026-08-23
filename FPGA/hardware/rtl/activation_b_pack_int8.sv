module activation_b_pack_int8 (
    input  logic clk, rst_n, pipeline_enable, start, transpose, source_row_major,
    input  logic [31:0] input_base_addr, output_base_addr,
    input  logic [15:0] reduction, outputs,
    output logic busy, done, error,
    output logic op_start, op_done, memory_stall, input_stall, output_stall,

    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [6:0] read_req_len,
    output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input logic read_data_valid, output logic read_data_ready,
    input logic [7:0] read_data, input logic read_data_last,

    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last,
    input logic memory_busy, memory_error
);
    typedef enum logic [3:0] {
        IDLE, SELECT_LANE, ADDR_PREP, ADDR_MULTIPLY, ADDR_FINAL, READ_REQ, READ_RECV, STORE,
        WRITE_LOW, WRITE_HIGH, WRITE_ARM, WRITE_WAIT, FINISH, FAIL
    } state_t;

    state_t state;
    logic transpose_r, source_row_major_r;
    logic [15:0] reduction_r, outputs_r, k_index, n_tile_base;
    logic [3:0] lane_index;
    logic [31:0] output_addr;
    logic [63:0] packed_vector_r;
    logic [15:0] source_row, source_column, source_features;
    logic [15:0] address_factor, address_column;
    logic [2:0] address_row_low;
    logic [31:0] address_product, read_addr_r;
    logic lane_valid;

    always_comb begin
        lane_valid = n_tile_base + lane_index < outputs_r;
        if (transpose_r) begin
            source_row = n_tile_base + lane_index;
            source_column = k_index;
            source_features = reduction_r;
        end else begin
            source_row = k_index;
            source_column = n_tile_base + lane_index;
            source_features = outputs_r;
        end
    end

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;
    assign read_req_valid = pipeline_enable && state == READ_REQ;
    assign read_req_addr = read_addr_r;
    assign read_req_len = 7'd1;
    assign read_req_wide = 1'b0;
    assign read_req_qword = 1'b0;
    assign read_req_allow_ahead = 1'b0;
    assign read_data_ready = pipeline_enable && state == READ_RECV;
    assign write_req_valid = pipeline_enable && (state == WRITE_LOW || state == WRITE_HIGH);
    assign write_req_addr = output_addr;
    assign write_req_data = state == WRITE_LOW ? packed_vector_r[31:0] : packed_vector_r[63:32];
    assign write_req_strb = 4'hf;
    assign write_req_last = state == WRITE_HIGH;
    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
                                   (read_data_ready && !read_data_valid) ||
                                   (write_req_valid && !write_req_ready));
    assign input_stall = state == READ_RECV && !read_data_valid;
    assign output_stall = (state == WRITE_LOW || state == WRITE_HIGH) && !write_req_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            transpose_r <= 1'b0; source_row_major_r <= 1'b0; reduction_r <= '0; outputs_r <= '0;
            k_index <= '0; n_tile_base <= '0; lane_index <= '0;
            output_addr <= '0; packed_vector_r <= '0; address_factor <= '0;
            address_column <= '0; address_row_low <= '0; address_product <= '0; read_addr_r <= '0;
        end else begin
            done <= 1'b0;
            if (memory_error) begin error <= 1'b1; state <= FAIL; end
            else if (pipeline_enable) unique case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (reduction == 0 || outputs == 0 || output_base_addr[1:0] != 0) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        transpose_r <= transpose; source_row_major_r <= source_row_major;
                        reduction_r <= reduction; outputs_r <= outputs;
                        k_index <= 0; n_tile_base <= 0; lane_index <= 0;
                        output_addr <= output_base_addr; packed_vector_r <= '0;
                        state <= SELECT_LANE;
                    end
                end
                SELECT_LANE: begin
                    if (lane_valid) state <= ADDR_PREP;
                    else begin packed_vector_r[lane_index*8 +: 8] <= 0; state <= STORE; end
                end
                ADDR_PREP: begin
                    address_factor <= source_row_major_r ? source_row : (source_row >> 3);
                    address_column <= source_column; address_row_low <= source_row[2:0];
                    state <= ADDR_MULTIPLY;
                end
                ADDR_MULTIPLY: begin address_product <= address_factor * source_features; state <= ADDR_FINAL; end
                ADDR_FINAL: begin
                    read_addr_r <= input_base_addr + (source_row_major_r
                        ? address_product + address_column
                        : ((address_product + address_column) << 3) + address_row_low);
                    state <= READ_REQ;
                end
                READ_REQ: if (read_req_valid && read_req_ready) state <= READ_RECV;
                READ_RECV: if (read_data_valid) begin
                    if (!read_data_last) begin error <= 1'b1; state <= FAIL; end
                    else begin packed_vector_r[lane_index*8 +: 8] <= read_data; state <= STORE; end
                end
                STORE: begin
                    if (lane_index == 7) begin lane_index <= 0; state <= WRITE_LOW; end
                    else begin lane_index <= lane_index + 1'b1; state <= SELECT_LANE; end
                end
                WRITE_LOW: if (write_req_valid && write_req_ready) begin output_addr <= output_addr + 4; state <= WRITE_HIGH; end
                WRITE_HIGH: if (write_req_valid && write_req_ready) begin
                    output_addr <= output_addr + 4; packed_vector_r <= '0;
                    if (k_index + 1'b1 == reduction_r) begin
                        k_index <= 0;
                        if (n_tile_base + 8 >= outputs_r) state <= WRITE_ARM;
                        else begin n_tile_base <= n_tile_base + 8; state <= SELECT_LANE; end
                    end else begin k_index <= k_index + 1'b1; state <= SELECT_LANE; end
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
