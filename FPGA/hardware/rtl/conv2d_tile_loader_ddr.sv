// Direct DDR im2col tile loader. Coordinate decode is sequential: no / or %
// operators are present on the DDR request path. Contiguous activation kernel
// rows and weight K-runs stream from cache-line AXI bursts.
module conv2d_tile_loader_ddr #(
    parameter int TILE_SIZE = 16
) (
    input logic clk, input logic rst_n, input logic start, input logic matrix_select, input logic conv_mode,
    input logic presliced_weights,
    input logic [31:0] base_addr, input logic [31:0] row_stride,
    input logic [31:0] input_base_addr, input logic [31:0] weight_base_addr,
    input logic [15:0] input_c, input logic [15:0] input_h, input logic [15:0] input_w,
    input logic [15:0] out_c, input logic [15:0] out_h, input logic [15:0] out_w,
    input logic [7:0] kernel_h, input logic [7:0] kernel_w, input logic [7:0] stride_h, input logic [7:0] stride_w,
    input logic [7:0] pad_h, input logic [7:0] pad_w, input logic [15:0] m0, input logic [15:0] k0, input logic [15:0] n0,
    input logic [4:0] tile_rows, input logic [4:0] tile_cols,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    output logic [4:0] read_req_len,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    input logic read_data_last,
    output logic cache_we, output logic cache_matrix, output logic [7:0] cache_addr, output logic [7:0] cache_wdata,
    output logic profile_address_phase, output logic profile_request_phase,
    output logic profile_receive_phase
);
    typedef enum logic [5:0] {
        IDLE, DECODE_M, DECODE_K_IC, DECODE_K_KH, BUILD,
        NORMAL_MUL, NORMAL_BASE, NORMAL_FINAL,
        COORD_MUL, COORD_ADD, COORD_CHECK,
        ADDR_SEED, ADDR_SEED_MUL, ADDR_ADD0, ADDR_MUL1_PREP,
        ADDR_MUL1_MUL0, ADDR_MUL1_MUL1, ADDR_MUL1_ADD01,
        ADDR_MUL1_MUL2, ADDR_MUL1_MUL3, ADDR_MUL1_ADD23, ADDR_MUL1_COMBINE, ADDR_ADD1,
        ADDR_MUL2_PREP, ADDR_MUL2_MUL0, ADDR_MUL2_MUL1, ADDR_MUL2_ADD01,
        ADDR_MUL2_MUL2, ADDR_MUL2_MUL3, ADDR_MUL2_ADD23, ADDR_MUL2_COMBINE,
        ADDR_ADD2, ADDR_BASE, BURST_CALC, BURST_PREP,
        PRESLICE_TILE_MUL, PRESLICE_TILE_BASE,
        REQUEST, RECEIVE, ADVANCE, FINISH, FAIL
    } state_t;
    state_t state;
    logic matrix_r, conv_r, presliced_r, pad_r;
    logic [4:0] row_r, col_r, rows_r, cols_r;
    logic [31:0] normal_base_r, normal_stride_r, request_addr_r;
    logic [15:0] in_c_r, in_h_r, in_w_r, out_c_r, out_w_r, n0_r;
    logic [7:0] kh_size_r, kw_size_r, sh_r, sw_r, ph_r, pw_r;
    logic [31:0] input_base_r, weight_base_r;
    logic [15:0] m_remaining, k_remaining, kernel_area_r;
    logic [15:0] row_oh_r, row_ow_r, base_oh_r, base_ow_r;
    logic [15:0] k_ic_r, k_kh_r, k_kw_r, base_ic_r, base_kh_r, base_kw_r;
    // Each address stage owns its output register. Reusing two temporaries
    // created a large state-controlled mux on addr_term0_r in routed logic.
    logic [31:0] normal_mul_r, normal_base_sum_r;
    (* use_dsp = "yes" *) logic [31:0] preslice_tile_index_r;
    logic [31:0] preslice_tile_base_r;
    logic [31:0] seed_product_r, addr_add0_r, addr_mul1_r, addr_add1_r;
    logic [31:0] addr_mul2_r, addr_add2_r;
    logic [15:0] addr_mul1_rhs_r;
    logic [23:0] addr_mul1_chunk0_r, addr_mul1_chunk1_r;
    logic [31:0] addr_mul1_pair_low_r, addr_mul1_pair_high_r;
    // Keep the input/weight address selection out of the seed multiplier's
    // operand and clock-enable cone.
    (* keep = "true" *) logic [15:0] addr_seed_lhs_r, addr_seed_rhs_r;
    logic signed [16:0] addr_ih_r, addr_iw_r;
    logic addr_weight_r;
    logic [15:0] addr_oc_r;
    logic [4:0] burst_len_calc_r, run_len_r, run_remaining_r;
    logic final_row_r, final_col_r;
    (* use_dsp = "yes" *) logic [31:0] coord_h_product_r, coord_w_product_r;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == REQUEST;
    assign read_req_addr = request_addr_r;
    assign read_req_len = run_len_r;
    assign read_data_ready = state == RECEIVE && !pad_r;
    assign cache_we = state == RECEIVE && (pad_r || read_data_valid);
    assign cache_matrix = matrix_r;
    assign cache_addr = row_r * TILE_SIZE + col_r;
    assign cache_wdata = pad_r ? 8'h00 : read_data;
    assign profile_request_phase = state == REQUEST;
    assign profile_receive_phase = state == RECEIVE;
    assign profile_address_phase = busy && state != REQUEST && state != RECEIVE;

    function automatic logic [4:0] min_run_len(
        input logic [16:0] first,
        input logic [16:0] second,
        input logic [16:0] third,
        input logic [16:0] fourth
    );
        logic [16:0] value;
        begin
            value = first > TILE_SIZE ? TILE_SIZE : first;
            if (second < value) value = second;
            if (third < value) value = third;
            if (fourth < value) value = fourth;
            min_run_len = value[4:0];
        end
    endfunction

    wire [16:0] burst_tile_left = (matrix_r && !presliced_r) ? rows_r - row_r : cols_r - col_r;
    wire [16:0] burst_kernel_left = matrix_r ? TILE_SIZE : kw_size_r - k_kw_r;
    wire [16:0] burst_input_left = matrix_r ? TILE_SIZE : in_w_r - addr_iw_r;
    wire [16:0] burst_line_left = 17'd64 - {11'd0, request_addr_r[5:0]};
    wire [4:0] burst_len = min_run_len(burst_tile_left, burst_kernel_left, burst_input_left, burst_line_left);

    task automatic increment_k;
        begin
            if (k_kw_r + 1 >= kw_size_r) begin
                k_kw_r <= '0;
                if (k_kh_r + 1 >= kh_size_r) begin k_kh_r <= '0; k_ic_r <= k_ic_r + 1'b1; end
                else k_kh_r <= k_kh_r + 1'b1;
            end else k_kw_r <= k_kw_r + 1'b1;
        end
    endtask
    task automatic increment_spatial;
        begin
            if (row_ow_r + 1 >= out_w_r) begin row_ow_r <= '0; row_oh_r <= row_oh_r + 1'b1; end
            else row_ow_r <= row_ow_r + 1'b1;
        end
    endtask

    always_ff @(posedge clk) begin
        integer ih, iw, oc;
        if (!rst_n) begin
            state <= IDLE; done <= 0; error <= 0; matrix_r <= 0; conv_r <= 0; presliced_r <= 0; pad_r <= 0; row_r <= 0; col_r <= 0;
            rows_r <= 0; cols_r <= 0; request_addr_r <= 0; m_remaining <= 0; k_remaining <= 0;
            row_oh_r <= 0; row_ow_r <= 0; base_oh_r <= 0; base_ow_r <= 0; k_ic_r <= 0; k_kh_r <= 0; k_kw_r <= 0; base_ic_r <= 0; base_kh_r <= 0; base_kw_r <= 0;
            normal_mul_r <= 0; normal_base_sum_r <= 0; preslice_tile_index_r <= 0; preslice_tile_base_r <= 0; seed_product_r <= 0; addr_add0_r <= 0; addr_mul1_r <= 0; addr_add1_r <= 0; addr_mul2_r <= 0; addr_add2_r <= 0;
            addr_mul1_rhs_r <= 0; addr_mul1_chunk0_r <= 0; addr_mul1_chunk1_r <= 0; addr_mul1_pair_low_r <= 0; addr_mul1_pair_high_r <= 0;
            addr_seed_lhs_r <= 0; addr_seed_rhs_r <= 0; addr_ih_r <= 0; addr_iw_r <= 0; addr_weight_r <= 0; addr_oc_r <= 0; coord_h_product_r <= 0; coord_w_product_r <= 0;
            burst_len_calc_r <= 5'd1; run_len_r <= 5'd1; run_remaining_r <= 5'd1; final_row_r <= 1'b0; final_col_r <= 1'b0;
        end else begin
            done <= 0;
            case (state)
                IDLE: if (start) begin
                    matrix_r<=matrix_select; conv_r<=conv_mode; presliced_r<=presliced_weights; row_r<=0; col_r<=0; rows_r<=tile_rows; cols_r<=tile_cols; error<=0;
                    normal_base_r<=base_addr; normal_stride_r<=row_stride; input_base_r<=input_base_addr; weight_base_r<=weight_base_addr;
                    in_c_r<=input_c; in_h_r<=input_h; in_w_r<=input_w; out_c_r<=out_c; out_w_r<=out_w; n0_r<=n0;
                    kh_size_r<=kernel_h; kw_size_r<=kernel_w; sh_r<=stride_h; sw_r<=stride_w; ph_r<=pad_h; pw_r<=pad_w; kernel_area_r<=kernel_h*kernel_w;
                    m_remaining<=m0; k_remaining<=k0;
                    base_oh_r<=0; base_ow_r<=0; row_oh_r<=0; row_ow_r<=0;
                    base_ic_r<=0; base_kh_r<=0; base_kw_r<=0; k_ic_r<=0; k_kh_r<=0; k_kw_r<=0;
                    if (tile_rows==0 || tile_rows>TILE_SIZE || tile_cols==0 || tile_cols>TILE_SIZE || (conv_mode && (out_w==0 || kernel_h==0 || kernel_w==0))) begin error<=1; state<=FAIL; end
                    else if (!conv_mode) state<=BUILD;
                    else if (matrix_select && presliced_weights) state<=PRESLICE_TILE_MUL;
                    else state<=DECODE_M;
                end
                PRESLICE_TILE_MUL: begin
                    // row_stride is K-tile count times 256 bytes. Multiply in
                    // tile units so this maps to one bounded 16x16 DSP stage.
                    preslice_tile_index_r <= {4'd0, n0_r[15:4]} * normal_stride_r[23:8];
                    state <= PRESLICE_TILE_BASE;
                end
                PRESLICE_TILE_BASE: begin
                    // k0 is tile aligned, so (k0 / 16) * 256 == k0 * 16.
                    preslice_tile_base_r <= weight_base_r + (preslice_tile_index_r << 8) + ({16'd0, k_remaining} << 4);
                    state <= BUILD;
                end
                DECODE_M: if (m_remaining >= out_w_r) begin m_remaining<=m_remaining-out_w_r; base_oh_r<=base_oh_r+1'b1; end
                          else begin base_ow_r<=m_remaining; row_oh_r<=base_oh_r; row_ow_r<=m_remaining; state<=DECODE_K_IC; end
                DECODE_K_IC: if (k_remaining >= kernel_area_r) begin k_remaining<=k_remaining-kernel_area_r; base_ic_r<=base_ic_r+1'b1; end
                             else state<=DECODE_K_KH;
                DECODE_K_KH: if (k_remaining >= kw_size_r) begin k_remaining<=k_remaining-kw_size_r; base_kh_r<=base_kh_r+1'b1; end
                             else begin base_kw_r<=k_remaining; k_ic_r<=base_ic_r; k_kh_r<=base_kh_r; k_kw_r<=k_remaining; state<=BUILD; end
                BUILD: begin
                    pad_r <= 0;
                    if (!conv_r) begin
                        state <= NORMAL_MUL;
                    end else if (!matrix_r) begin
                        addr_weight_r <= 1'b0;
                        state <= COORD_MUL;
                    end else if (presliced_r) begin
                        request_addr_r <= preslice_tile_base_r + ({27'd0, row_r} << 4) + col_r;
                        state <= BURST_CALC;
                    end else begin
                        addr_weight_r <= 1'b1;
                        addr_oc_r <= n0_r + col_r;
                        addr_ih_r <= $signed({1'b0, k_kh_r});
                        addr_iw_r <= $signed({1'b0, k_kw_r});
                        state <= ADDR_SEED;
                    end
                end
                // Every address arithmetic level is registered.  Keeping the
                // coordinate, Horner multiply, and add stages separate avoids
                // a long carry chain from the tile indices into the request.
                NORMAL_MUL: begin
                    normal_mul_r <= row_r * normal_stride_r;
                    state <= NORMAL_BASE;
                end
                NORMAL_BASE: begin
                    normal_base_sum_r <= normal_base_r + normal_mul_r;
                    state <= NORMAL_FINAL;
                end
                NORMAL_FINAL: begin
                    request_addr_r <= normal_base_sum_r + col_r;
                    run_len_r <= 5'd1;
                    run_remaining_r <= 5'd1;
                    state <= REQUEST;
                end
                COORD_MUL: begin
                    coord_h_product_r <= row_oh_r * sh_r;
                    coord_w_product_r <= row_ow_r * sw_r;
                    state <= COORD_ADD;
                end
                COORD_ADD: begin
                    addr_ih_r <= $signed({1'b0, coord_h_product_r}) + $signed({1'b0, k_kh_r}) - $signed({1'b0, ph_r});
                    addr_iw_r <= $signed({1'b0, coord_w_product_r}) + $signed({1'b0, k_kw_r}) - $signed({1'b0, pw_r});
                    state <= COORD_CHECK;
                end
                COORD_CHECK: begin
                    if (k_ic_r >= in_c_r || addr_ih_r < 0 || addr_ih_r >= $signed({1'b0, in_h_r}) || addr_iw_r < 0 || addr_iw_r >= $signed({1'b0, in_w_r})) begin
                        pad_r <= 1'b1;
                        run_remaining_r <= 5'd1;
                        state <= RECEIVE;
                    end else begin
                        state <= ADDR_SEED;
                    end
                end
                ADDR_SEED: begin
                    if (addr_weight_r && (k_ic_r >= in_c_r || addr_oc_r >= out_c_r)) begin
                        pad_r <= 1'b1;
                        run_remaining_r <= 5'd1;
                        state <= RECEIVE;
                    end else begin
                        if (addr_weight_r) begin
                            addr_seed_lhs_r <= addr_oc_r;
                            addr_seed_rhs_r <= in_c_r;
                        end else begin
                            addr_seed_lhs_r <= k_ic_r;
                            addr_seed_rhs_r <= in_h_r;
                        end
                        state <= ADDR_SEED_MUL;
                    end
                end
                ADDR_SEED_MUL: begin
                    seed_product_r <= addr_seed_lhs_r * addr_seed_rhs_r;
                    state <= ADDR_ADD0;
                end
                ADDR_ADD0: begin
                    if (addr_weight_r) begin
                        addr_add0_r <= seed_product_r + k_ic_r;
                    end else begin
                        addr_add0_r <= seed_product_r + addr_ih_r;
                    end
                    state <= ADDR_MUL1_PREP;
                end
                // Register the selected factor, then decompose the 32x16
                // Horner product into four independent byte products.
                ADDR_MUL1_PREP: begin
                    addr_mul1_rhs_r <= addr_weight_r ? {8'd0, kh_size_r} : in_w_r;
                    state <= ADDR_MUL1_MUL0;
                end
                ADDR_MUL1_MUL0: begin addr_mul1_chunk0_r <= addr_add0_r[7:0] * addr_mul1_rhs_r; state <= ADDR_MUL1_MUL1; end
                ADDR_MUL1_MUL1: begin addr_mul1_chunk1_r <= addr_add0_r[15:8] * addr_mul1_rhs_r; state <= ADDR_MUL1_ADD01; end
                ADDR_MUL1_ADD01: begin addr_mul1_pair_low_r <= {8'd0, addr_mul1_chunk0_r} + ({8'd0, addr_mul1_chunk1_r} << 8); state <= ADDR_MUL1_MUL2; end
                ADDR_MUL1_MUL2: begin addr_mul1_chunk0_r <= addr_add0_r[23:16] * addr_mul1_rhs_r; state <= ADDR_MUL1_MUL3; end
                ADDR_MUL1_MUL3: begin addr_mul1_chunk1_r <= addr_add0_r[31:24] * addr_mul1_rhs_r; state <= ADDR_MUL1_ADD23; end
                ADDR_MUL1_ADD23: begin addr_mul1_pair_high_r <= {8'd0, addr_mul1_chunk0_r} + ({8'd0, addr_mul1_chunk1_r} << 8); state <= ADDR_MUL1_COMBINE; end
                ADDR_MUL1_COMBINE: begin addr_mul1_r <= addr_mul1_pair_low_r + (addr_mul1_pair_high_r << 16); state <= ADDR_ADD1; end
                ADDR_ADD1: begin
                    if (addr_weight_r) begin
                        addr_add1_r <= addr_mul1_r + addr_ih_r;
                        state <= ADDR_MUL2_PREP;
                    end else begin
                        addr_add1_r <= addr_mul1_r + addr_iw_r;
                        state <= ADDR_BASE;
                    end
                end
                ADDR_MUL2_PREP: begin addr_mul1_rhs_r <= {8'd0, kw_size_r}; state <= ADDR_MUL2_MUL0; end
                ADDR_MUL2_MUL0: begin addr_mul1_chunk0_r <= addr_add1_r[7:0] * addr_mul1_rhs_r; state <= ADDR_MUL2_MUL1; end
                ADDR_MUL2_MUL1: begin addr_mul1_chunk1_r <= addr_add1_r[15:8] * addr_mul1_rhs_r; state <= ADDR_MUL2_ADD01; end
                ADDR_MUL2_ADD01: begin addr_mul1_pair_low_r <= {8'd0, addr_mul1_chunk0_r} + ({8'd0, addr_mul1_chunk1_r} << 8); state <= ADDR_MUL2_MUL2; end
                ADDR_MUL2_MUL2: begin addr_mul1_chunk0_r <= addr_add1_r[23:16] * addr_mul1_rhs_r; state <= ADDR_MUL2_MUL3; end
                ADDR_MUL2_MUL3: begin addr_mul1_chunk1_r <= addr_add1_r[31:24] * addr_mul1_rhs_r; state <= ADDR_MUL2_ADD23; end
                ADDR_MUL2_ADD23: begin addr_mul1_pair_high_r <= {8'd0, addr_mul1_chunk0_r} + ({8'd0, addr_mul1_chunk1_r} << 8); state <= ADDR_MUL2_COMBINE; end
                ADDR_MUL2_COMBINE: begin addr_mul2_r <= addr_mul1_pair_low_r + (addr_mul1_pair_high_r << 16); state <= ADDR_ADD2; end
                ADDR_ADD2: begin
                    addr_add2_r <= addr_mul2_r + addr_iw_r;
                    state <= ADDR_BASE;
                end
                ADDR_BASE: begin
                    request_addr_r <= (addr_weight_r ? weight_base_r : input_base_r) + (addr_weight_r ? addr_add2_r : addr_add1_r);
                    state <= BURST_CALC;
                end
                BURST_CALC: begin
                    // Isolate coordinate/line limit arithmetic from the FSM,
                    // error flag, and duplicated run counters.
                    burst_len_calc_r <= burst_len;
                    state <= BURST_PREP;
                end
                BURST_PREP: begin
                    run_len_r <= burst_len_calc_r;
                    run_remaining_r <= burst_len_calc_r;
                    if (burst_len_calc_r == 0) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        state <= REQUEST;
                    end
                end
                REQUEST: begin
                    if (read_req_ready) state <= RECEIVE;
                end
                RECEIVE: begin
                    if (pad_r || read_data_valid) begin
                        // Stream all interior burst beats without bubbles. On
                        // the final beat, register boundary comparisons so the
                        // coordinate counters cannot feed the FSM state cone.
                        if (!pad_r && (read_data_last != (run_remaining_r == 1))) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else if (conv_r && run_remaining_r > 1) begin
                            run_remaining_r <= run_remaining_r - 1'b1;
                            if (matrix_r && presliced_r) col_r <= col_r + 1'b1;
                            else increment_k();
                            if (matrix_r && !presliced_r) row_r <= row_r + 1'b1;
                            else if (!(matrix_r && presliced_r)) col_r <= col_r + 1'b1;
                        end else begin
                            final_row_r <= row_r == rows_r - 1'b1;
                            final_col_r <= col_r == cols_r - 1'b1;
                            state <= ADVANCE;
                        end
                    end
                end
                ADVANCE: begin
                    if (final_row_r && final_col_r) begin
                        state <= FINISH;
                    end else if (!conv_r) begin
                        if (final_col_r) begin
                            col_r <= '0;
                            row_r <= row_r + 1'b1;
                            if (matrix_r) increment_k();
                        end else begin
                            col_r <= col_r + 1'b1;
                            if (matrix_r) increment_k();
                        end
                        state <= BUILD;
                    end else if (!matrix_r) begin
                        if (final_col_r) begin
                            col_r <= '0;
                            row_r <= row_r + 1'b1;
                            increment_spatial();
                            k_ic_r <= base_ic_r;
                            k_kh_r <= base_kh_r;
                            k_kw_r <= base_kw_r;
                        end else begin
                            col_r <= col_r + 1'b1;
                            increment_k();
                        end
                        state <= BUILD;
                    end else if (presliced_r) begin
                        if (final_col_r) begin
                            col_r <= '0;
                            row_r <= row_r + 1'b1;
                        end else col_r <= col_r + 1'b1;
                        state <= BUILD;
                    end else if (final_row_r) begin
                        col_r <= col_r + 1'b1;
                        row_r <= '0;
                        k_ic_r <= base_ic_r;
                        k_kh_r <= base_kh_r;
                        k_kw_r <= base_kw_r;
                        state <= BUILD;
                    end else begin
                        row_r <= row_r + 1'b1;
                        increment_k();
                        state <= BUILD;
                    end
                end
                FINISH: begin done<=1; state<=IDLE; end
                FAIL: begin done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
