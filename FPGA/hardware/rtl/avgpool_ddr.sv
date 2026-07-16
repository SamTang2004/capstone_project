// NCHW signed INT8 AveragePool. GlobalAveragePool is the kernel=HxW,
// stride=1, out_h=out_w=1 configuration. Padding samples are excluded.
module avgpool_ddr (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] input_base_addr, input logic [31:0] output_base_addr,
    input logic [15:0] n, input logic [15:0] channels, input logic [15:0] input_h, input logic [15:0] input_w,
    input logic [15:0] output_h, input logic [15:0] output_w,
    input logic [7:0] kernel_h, input logic [7:0] kernel_w, input logic [7:0] stride_h, input logic [7:0] stride_w,
    input logic [7:0] pad_h, input logic [7:0] pad_w,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    output logic write_req_valid, input logic write_req_ready, output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data, output logic [3:0] write_req_strb
);
    typedef enum logic [4:0] {
        IDLE, WINDOW, COORD_MUL, COORD_ADD, COORD_CHECK,
        ADDR_PLANE_MUL, ADDR_PLANE_ADD,
        ADDR_ROW_MUL0, ADDR_ROW_MUL1, ADDR_ROW_ADD01, ADDR_ROW_MUL2,
        ADDR_ROW_MUL3, ADDR_ROW_ADD23, ADDR_ROW_COMBINE, ADDR_ROW_ADD,
        ADDR_LINEAR_MUL0, ADDR_LINEAR_MUL1, ADDR_LINEAR_ADD01, ADDR_LINEAR_MUL2,
        ADDR_LINEAR_MUL3, ADDR_LINEAR_ADD23, ADDR_LINEAR_COMBINE, ADDR_LINEAR_ADD, ADDR_BASE,
        READ_REQ, READ_WAIT, DIV_INIT, DIV_LOOP, PACK, WRITE, FINISH, FAIL
    } state_t;
    state_t state;
    logic [31:0] in_r, out_r, out_index_r, packed_r, write_addr_r;
    logic [31:0] addr_plane_r, addr_row_r, addr_linear_r, sample_addr_r;
    logic [23:0] addr_chunk0_r, addr_chunk1_r;
    logic [31:0] addr_pair_low_r, addr_pair_high_r;
    (* use_dsp = "yes" *) logic [31:0] coord_y_product_r, coord_x_product_r;
    logic [15:0] n_r, c_r, ih_r, iw_r, oh_r, ow_r;
    logic [7:0] kh_r, kw_r, sh_r, sw_r, ph_r, pw_r;
    logic [15:0] ni, ci, oy, ox;
    logic [7:0] ky, kx;
    logic signed [31:0] sum_r;
    logic [15:0] count_r;
    logic [31:0] dividend_r, divisor_r, quotient_r;
    logic negative_r, write_last_r;
    logic [3:0] write_strb_r;
    logic signed [31:0] sample_y, sample_x;
    logic sample_valid;
    logic last_output;

    always_comb begin
        last_output = ni + 1 >= n_r && ci + 1 >= c_r && oy + 1 >= oh_r && ox + 1 >= ow_r;
    end

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = sample_addr_r;
    // A response is consumed only after its request has been accepted.  This
    // prevents a registered memory port's previous valid pulse being reused.
    assign read_data_ready = state == READ_WAIT;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = packed_r;
    assign write_req_strb = write_strb_r;

    task automatic advance_window;
        begin
            if (kx + 1 >= kw_r) begin
                kx <= 0;
                if (ky + 1 >= kh_r) begin
                    ky <= 0;
                    state <= DIV_INIT;
                end else begin ky <= ky + 1'b1; state <= WINDOW; end
            end else begin kx <= kx + 1'b1; state <= WINDOW; end
        end
    endtask

    task automatic advance_output;
        begin
            if (ox + 1 >= ow_r) begin
                ox <= 0;
                if (oy + 1 >= oh_r) begin
                    oy <= 0;
                    if (ci + 1 >= c_r) begin ci <= 0; ni <= ni + 1'b1; end
                    else ci <= ci + 1'b1;
                end else oy <= oy + 1'b1;
            end else ox <= ox + 1'b1;
        end
    endtask

    always_ff @(posedge clk) begin
        logic signed [31:0] final_sum;
        logic [15:0] final_count;
        logic [31:0] next_packed;
        logic signed [31:0] average;
        if (!rst_n) begin
            state <= IDLE; done <= 0; error <= 0; in_r <= 0; out_r <= 0; out_index_r <= 0; packed_r <= 0; write_addr_r <= 0; write_strb_r <= 0; write_last_r <= 0;
            addr_plane_r <= 0; addr_row_r <= 0; addr_linear_r <= 0; sample_addr_r <= 0;
            addr_chunk0_r <= 0; addr_chunk1_r <= 0; addr_pair_low_r <= 0; addr_pair_high_r <= 0;
            coord_y_product_r <= 0; coord_x_product_r <= 0; sample_y <= 0; sample_x <= 0; sample_valid <= 0;
            n_r <= 0; c_r <= 0; ih_r <= 0; iw_r <= 0; oh_r <= 0; ow_r <= 0; kh_r <= 0; kw_r <= 0; sh_r <= 0; sw_r <= 0; ph_r <= 0; pw_r <= 0;
            ni <= 0; ci <= 0; oy <= 0; ox <= 0; ky <= 0; kx <= 0; sum_r <= 0; count_r <= 0; dividend_r <= 0; divisor_r <= 0; quotient_r <= 0; negative_r <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: if (start) begin
                    in_r <= input_base_addr; out_r <= output_base_addr; n_r <= n; c_r <= channels; ih_r <= input_h; iw_r <= input_w; oh_r <= output_h; ow_r <= output_w;
                    kh_r <= kernel_h; kw_r <= kernel_w; sh_r <= stride_h; sw_r <= stride_w; ph_r <= pad_h; pw_r <= pad_w;
                    ni <= 0; ci <= 0; oy <= 0; ox <= 0; ky <= 0; kx <= 0; out_index_r <= 0; packed_r <= 0; error <= 0;
                    addr_plane_r <= 0; addr_row_r <= 0; addr_linear_r <= 0; sample_addr_r <= 0;
                    addr_chunk0_r <= 0; addr_chunk1_r <= 0; addr_pair_low_r <= 0; addr_pair_high_r <= 0;
                    coord_y_product_r <= 0; coord_x_product_r <= 0; sample_y <= 0; sample_x <= 0; sample_valid <= 0;
                    if (n == 0 || channels == 0 || input_h == 0 || input_w == 0 || output_h == 0 || output_w == 0 || kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0) begin error <= 1; state <= FAIL; end else state <= WINDOW;
                end
                WINDOW: begin
                    if (ky == 0 && kx == 0) begin sum_r <= 0; count_r <= 0; end
                    state <= COORD_MUL;
                end
                // Multiply, add, and bounds decisions are independently
                // registered so runtime NCHW dimensions never form a long
                // carry chain into a memory request.
                COORD_MUL: begin
                    coord_y_product_r <= oy * sh_r;
                    coord_x_product_r <= ox * sw_r;
                    state <= COORD_ADD;
                end
                COORD_ADD: begin
                    sample_y <= $signed({1'b0, coord_y_product_r}) + $signed({1'b0, ky}) - $signed({1'b0, ph_r});
                    sample_x <= $signed({1'b0, coord_x_product_r}) + $signed({1'b0, kx}) - $signed({1'b0, pw_r});
                    state <= COORD_CHECK;
                end
                COORD_CHECK: begin
                    sample_valid <= sample_y >= 0 && sample_x >= 0 && sample_y < $signed({1'b0, ih_r}) && sample_x < $signed({1'b0, iw_r});
                    if (sample_y >= 0 && sample_x >= 0 && sample_y < $signed({1'b0, ih_r}) && sample_x < $signed({1'b0, iw_r})) state <= ADDR_PLANE_MUL;
                    else advance_window();
                end
                ADDR_PLANE_MUL: begin
                    addr_plane_r <= ni * c_r;
                    state <= ADDR_PLANE_ADD;
                end
                ADDR_PLANE_ADD: begin
                    addr_plane_r <= addr_plane_r + ci;
                    state <= ADDR_ROW_MUL0;
                end
                // The DSP array consumes every DSP48. Decompose each 32x16
                // address multiply into four registered 8x16 products and
                // two levels of registered additions so pooling stays out of
                // the accelerator's critical timing cone.
                ADDR_ROW_MUL0: begin addr_chunk0_r <= addr_plane_r[7:0] * ih_r; state <= ADDR_ROW_MUL1; end
                ADDR_ROW_MUL1: begin addr_chunk1_r <= addr_plane_r[15:8] * ih_r; state <= ADDR_ROW_ADD01; end
                ADDR_ROW_ADD01: begin addr_pair_low_r <= {8'd0, addr_chunk0_r} + ({8'd0, addr_chunk1_r} << 8); state <= ADDR_ROW_MUL2; end
                ADDR_ROW_MUL2: begin addr_chunk0_r <= addr_plane_r[23:16] * ih_r; state <= ADDR_ROW_MUL3; end
                ADDR_ROW_MUL3: begin addr_chunk1_r <= addr_plane_r[31:24] * ih_r; state <= ADDR_ROW_ADD23; end
                ADDR_ROW_ADD23: begin addr_pair_high_r <= {8'd0, addr_chunk0_r} + ({8'd0, addr_chunk1_r} << 8); state <= ADDR_ROW_COMBINE; end
                ADDR_ROW_COMBINE: begin addr_row_r <= addr_pair_low_r + (addr_pair_high_r << 16); state <= ADDR_ROW_ADD; end
                ADDR_ROW_ADD: begin
                    addr_row_r <= addr_row_r + sample_y;
                    state <= ADDR_LINEAR_MUL0;
                end
                ADDR_LINEAR_MUL0: begin addr_chunk0_r <= addr_row_r[7:0] * iw_r; state <= ADDR_LINEAR_MUL1; end
                ADDR_LINEAR_MUL1: begin addr_chunk1_r <= addr_row_r[15:8] * iw_r; state <= ADDR_LINEAR_ADD01; end
                ADDR_LINEAR_ADD01: begin addr_pair_low_r <= {8'd0, addr_chunk0_r} + ({8'd0, addr_chunk1_r} << 8); state <= ADDR_LINEAR_MUL2; end
                ADDR_LINEAR_MUL2: begin addr_chunk0_r <= addr_row_r[23:16] * iw_r; state <= ADDR_LINEAR_MUL3; end
                ADDR_LINEAR_MUL3: begin addr_chunk1_r <= addr_row_r[31:24] * iw_r; state <= ADDR_LINEAR_ADD23; end
                ADDR_LINEAR_ADD23: begin addr_pair_high_r <= {8'd0, addr_chunk0_r} + ({8'd0, addr_chunk1_r} << 8); state <= ADDR_LINEAR_COMBINE; end
                ADDR_LINEAR_COMBINE: begin addr_linear_r <= addr_pair_low_r + (addr_pair_high_r << 16); state <= ADDR_LINEAR_ADD; end
                ADDR_LINEAR_ADD: begin
                    addr_linear_r <= addr_linear_r + sample_x;
                    state <= ADDR_BASE;
                end
                ADDR_BASE: begin
                    sample_addr_r <= in_r + addr_linear_r;
                    state <= READ_REQ;
                end
                READ_REQ: if (read_req_ready) state <= READ_WAIT;
                READ_WAIT: if (read_data_valid) begin
                    // Keep the DDR byte's sign extension explicit at the
                    // accumulator boundary.
                    final_sum = sum_r + $signed({{24{read_data[7]}}, read_data}); final_count = count_r + 1'b1; sum_r <= final_sum; count_r <= final_count;
                    if (ky + 1 >= kh_r && kx + 1 >= kw_r) state <= DIV_INIT; else advance_window();
                end
                DIV_INIT: begin
                    // The final read enters this state directly, so reset the
                    // kernel cursor before the next output window begins.
                    ky <= 0;
                    kx <= 0;
                    if (count_r == 0) begin error <= 1; state <= FAIL; end
                    else begin negative_r <= sum_r < 0; dividend_r <= sum_r < 0 ? -sum_r : sum_r; divisor_r <= count_r; quotient_r <= 0; state <= DIV_LOOP; end
                end
                DIV_LOOP: if (dividend_r >= divisor_r) begin dividend_r <= dividend_r - divisor_r; quotient_r <= quotient_r + 1'b1; end
                else begin average = negative_r ? -$signed(quotient_r) : $signed(quotient_r); next_packed = packed_r; next_packed[out_index_r[1:0]*8 +: 8] = average[7:0]; packed_r <= next_packed;
                    if (out_index_r[1:0] == 2'd3 || last_output) begin write_addr_r <= out_r + (out_index_r & 32'hffff_fffc); write_strb_r <= out_index_r[1:0] == 2'd3 ? 4'hf : ((4'b0001 << (out_index_r[1:0] + 1'b1)) - 1'b1); write_last_r <= last_output; packed_r <= next_packed; state <= WRITE; end
                    else begin out_index_r <= out_index_r + 1'b1; advance_output(); state <= WINDOW; end
                end
                WRITE: if (write_req_ready) begin packed_r <= 0; if (write_last_r) state <= FINISH; else begin out_index_r <= out_index_r + 1'b1; advance_output(); state <= WINDOW; end end
                FINISH: begin done <= 1; state <= IDLE; end
                FAIL: begin done <= 1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
