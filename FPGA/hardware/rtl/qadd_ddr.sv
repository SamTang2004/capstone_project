// Equal-domain INT8 residual add: q_out = saturate(q_a + q_b - zero_point).
module qadd_ddr (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] a_base_addr, input logic [31:0] b_base_addr, input logic [31:0] output_base_addr,
    input logic [31:0] element_count, input logic signed [31:0] output_zero_point,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    output logic write_req_valid, input logic write_req_ready, output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data, output logic [3:0] write_req_strb
);
    typedef enum logic [3:0] { IDLE, A_REQ, A_WAIT, B_REQ, B_WAIT, PACK, WRITE, FINISH, FAIL } state_t;
    state_t state;
    logic [31:0] a_r, b_r, out_r, count_r, index_r, packed_r, write_addr_r;
    logic [3:0] write_strb_r;
    logic signed [31:0] zp_r;
    logic signed [7:0] a_value_r;
    function automatic logic signed [7:0] sat_i8(input logic signed [31:0] value);
        if (value < -32'sd128) sat_i8 = -8'sd128;
        else if (value > 32'sd127) sat_i8 = 8'sd127;
        else sat_i8 = value[7:0];
    endfunction
    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == A_REQ || state == B_REQ;
    assign read_req_addr = (state == A_REQ ? a_r : b_r) + index_r;
    assign read_data_ready = state == A_WAIT || state == B_WAIT || ((state == A_REQ || state == B_REQ) && read_data_valid);
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = packed_r;
    assign write_req_strb = write_strb_r;
    always_ff @(posedge clk) begin
        logic [31:0] next_packed;
        logic signed [31:0] sum;
        if (!rst_n) begin state <= IDLE; done <= 0; error <= 0; a_r <= 0; b_r <= 0; out_r <= 0; count_r <= 0; index_r <= 0; packed_r <= 0; write_addr_r <= 0; write_strb_r <= 0; zp_r <= 0; a_value_r <= 0; end
        else begin
            done <= 0;
            case (state)
                IDLE: if (start) begin
                    a_r <= a_base_addr; b_r <= b_base_addr; out_r <= output_base_addr; count_r <= element_count; zp_r <= output_zero_point; index_r <= 0; packed_r <= 0; error <= 0;
                    if (element_count == 0 || a_base_addr[1:0] != 0 || b_base_addr[1:0] != 0 || output_base_addr[1:0] != 0) begin error <= 1; state <= FAIL; end else state <= A_REQ;
                end
                A_REQ: if (read_data_valid) begin a_value_r <= read_data; state <= B_REQ; end else if (read_req_ready) state <= A_WAIT;
                A_WAIT: if (read_data_valid) begin a_value_r <= read_data; state <= B_REQ; end
                B_REQ: if (read_data_valid) begin
                    sum = a_value_r + $signed(read_data) - zp_r; next_packed = packed_r; next_packed[index_r[1:0]*8 +: 8] = sat_i8(sum); packed_r <= next_packed;
                    if (index_r[1:0] == 2'd3 || index_r + 1 >= count_r) begin write_addr_r <= out_r + (index_r & 32'hffff_fffc); write_strb_r <= index_r[1:0] == 2'd3 ? 4'hf : ((4'b0001 << (index_r[1:0] + 1'b1)) - 1'b1); packed_r <= next_packed; state <= WRITE; end else begin index_r <= index_r + 1'b1; state <= A_REQ; end
                end else if (read_req_ready) state <= B_WAIT;
                B_WAIT: if (read_data_valid) begin
                    sum = a_value_r + $signed(read_data) - zp_r; next_packed = packed_r; next_packed[index_r[1:0]*8 +: 8] = sat_i8(sum); packed_r <= next_packed;
                    if (index_r[1:0] == 2'd3 || index_r + 1 >= count_r) begin write_addr_r <= out_r + (index_r & 32'hffff_fffc); write_strb_r <= index_r[1:0] == 2'd3 ? 4'hf : ((4'b0001 << (index_r[1:0] + 1'b1)) - 1'b1); state <= WRITE; end else begin index_r <= index_r + 1'b1; state <= A_REQ; end
                end
                WRITE: if (write_req_ready) begin packed_r <= 0; if (index_r + 1 >= count_r) state <= FINISH; else begin index_r <= index_r + 1'b1; state <= A_REQ; end end
                FINISH: begin done <= 1; state <= IDLE; end
                FAIL: begin done <= 1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
