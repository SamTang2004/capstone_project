// Incremental im2col activation descriptor generator. The expensive tensor
// products are paid once during setup; RUN advances with additions and nested
// counters so one burst descriptor can be accepted every clock.
(* use_dsp = "no" *) module conv2d_activation_address_gen #(
    parameter int TILE_SIZE = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [31:0] input_base_addr,
    input  logic [15:0] input_c,
    input  logic [15:0] input_h,
    input  logic [15:0] input_w,
    input  logic [15:0] out_w,
    input  logic [7:0]  kernel_h,
    input  logic [7:0]  kernel_w,
    input  logic [7:0]  stride_h,
    input  logic [7:0]  stride_w,
    input  logic [7:0]  pad_h,
    input  logic [7:0]  pad_w,
    input  logic [15:0] m0,
    input  logic [15:0] k0,
    input  logic [4:0]  tile_rows,
    input  logic [4:0]  tile_cols,
    output logic        busy,
    output logic        done,
    output logic        error,
    output logic        descriptor_valid,
    input  logic        descriptor_ready,
    output logic [31:0] descriptor_addr,
    output logic [4:0]  descriptor_len,
    output logic        descriptor_pad,
    output logic [4:0]  descriptor_row,
    output logic [4:0]  descriptor_col,
    output logic        descriptor_last
);
    typedef enum logic [4:0] {
        IDLE, START_KERNEL_AREA, WAIT_KERNEL_AREA,
        DECODE_M, DECODE_K_IC, DECODE_K_KH,
        START_OH, WAIT_OH, START_OW, WAIT_OW,
        START_CHANNEL_STRIDE, WAIT_CHANNEL_STRIDE,
        START_ROW_HSTEP, WAIT_ROW_HSTEP, START_ROW_WSPAN, WAIT_ROW_WSPAN,
        START_KH_SPAN, WAIT_KH_SPAN, START_K_SEED, WAIT_K_SEED,
        INIT_K_ADD, START_K_OFFSET, WAIT_K_OFFSET,
        START_SPATIAL, WAIT_SPATIAL, INIT_ADDRESS,
        CHECK_DESCRIPTOR, PREP_DESCRIPTOR, RUN, ADVANCE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [31:0] base_r;
    logic [15:0] in_c_r, in_h_r, in_w_r, out_w_r;
    logic [7:0] kh_size_r, kw_size_r, sh_r, sw_r, ph_r, pw_r;
    logic [4:0] rows_r, cols_r, row_r, col_r;
    logic [15:0] m_remaining_r, k_remaining_r, kernel_area_r;
    logic [15:0] oh_r, ow_r, base_ic_r, base_kh_r, base_kw_r;
    logic [15:0] ic_r, kh_r, kw_r;
    logic signed [16:0] spatial_ih_r, spatial_iw_r;
    logic [31:0] channel_stride_r, row_hstep_r, row_wspan_r;
    logic [31:0] channel_wrap_delta_r;
    logic [31:0] k_seed_r, k_add_r, k_offset_r;
    logic signed [32:0] spatial_product_r;
    logic signed [32:0] address_r, row_start_address_r;
    logic mul_start_r, mul_signed_r, mul_busy, mul_done;
    logic [31:0] mul_lhs_r;
    logic [15:0] mul_rhs_r;
    logic [47:0] mul_product;
    logic [5:0] accepted_len_r;
    logic accepted_last_r, checked_pad_r;
    logic [16:0] checked_tile_left_r, checked_kernel_left_r;
    logic [16:0] checked_input_left_r, checked_line_left_r;
    logic [4:0] prepared_len_r;

    logic signed [16:0] current_ih, current_iw;
    logic current_valid;
    logic [16:0] tile_left, kernel_left, input_left, line_left;

    lut_mul32x16_pipeline setup_multiplier (
        .clk(clk), .rst_n(rst_n), .start(mul_start_r), .signed_mode(mul_signed_r),
        .lhs(mul_lhs_r), .rhs(mul_rhs_r), .busy(mul_busy), .done(mul_done),
        .product(mul_product)
    );

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

    always_comb begin
        current_ih = spatial_ih_r + $signed({1'b0, kh_r});
        current_iw = spatial_iw_r + $signed({1'b0, kw_r});
        current_valid = ic_r < in_c_r && current_ih >= 0 &&
                        current_ih < $signed({1'b0, in_h_r}) &&
                        current_iw >= 0 && current_iw < $signed({1'b0, in_w_r});
        tile_left = {12'd0, cols_r} - {12'd0, col_r};
        kernel_left = {9'd0, kw_size_r} - {1'b0, kw_r};
        // Invalid coordinates are masked by checked_pad_r. Keeping validity
        // out of this distance cone lets run length and padding register in
        // parallel instead of serializing a wide bounds comparison into min.
        input_left = {1'b0, in_w_r} - current_iw;
        line_left = 17'd64 - {11'd0, address_r[5:0]};
    end

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign done = state == FINISH || state == FAIL;
    assign descriptor_valid = state == RUN;
    assign descriptor_addr = address_r[31:0];
    assign descriptor_len = checked_pad_r ? 5'd1 : prepared_len_r;
    assign descriptor_pad = checked_pad_r;
    assign descriptor_row = row_r;
    assign descriptor_col = col_r;
    assign descriptor_last = row_r == rows_r - 1'b1 &&
                             col_r + descriptor_len >= cols_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            error <= 1'b0;
            base_r <= '0;
            in_c_r <= '0; in_h_r <= '0; in_w_r <= '0; out_w_r <= '0;
            kh_size_r <= '0; kw_size_r <= '0; sh_r <= '0; sw_r <= '0; ph_r <= '0; pw_r <= '0;
            rows_r <= '0; cols_r <= '0; row_r <= '0; col_r <= '0;
            m_remaining_r <= '0; k_remaining_r <= '0; kernel_area_r <= '0;
            oh_r <= '0; ow_r <= '0; base_ic_r <= '0; base_kh_r <= '0; base_kw_r <= '0;
            ic_r <= '0; kh_r <= '0; kw_r <= '0;
            spatial_ih_r <= '0; spatial_iw_r <= '0;
            channel_stride_r <= '0; row_hstep_r <= '0; row_wspan_r <= '0;
            channel_wrap_delta_r <= '0;
            k_seed_r <= '0; k_add_r <= '0; k_offset_r <= '0;
            spatial_product_r <= '0; address_r <= '0; row_start_address_r <= '0;
            mul_start_r <= 1'b0; mul_signed_r <= 1'b0; mul_lhs_r <= '0; mul_rhs_r <= '0;
            accepted_len_r <= '0;
            accepted_last_r <= 1'b0; checked_pad_r <= 1'b0;
            checked_tile_left_r <= '0; checked_kernel_left_r <= '0;
            checked_input_left_r <= '0; checked_line_left_r <= '0; prepared_len_r <= 5'd1;
        end else begin
            mul_start_r <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    base_r <= input_base_addr;
                    in_c_r <= input_c; in_h_r <= input_h; in_w_r <= input_w; out_w_r <= out_w;
                    kh_size_r <= kernel_h; kw_size_r <= kernel_w;
                    sh_r <= stride_h; sw_r <= stride_w; ph_r <= pad_h; pw_r <= pad_w;
                    rows_r <= tile_rows; cols_r <= tile_cols; row_r <= '0; col_r <= '0;
                    m_remaining_r <= m0; k_remaining_r <= k0;
                    oh_r <= '0; ow_r <= '0; base_ic_r <= '0; base_kh_r <= '0; base_kw_r <= '0;
                    if (tile_rows == 0 || tile_rows > TILE_SIZE || tile_cols == 0 ||
                        tile_cols > TILE_SIZE || input_h == 0 || input_w == 0 ||
                        out_w == 0 || kernel_h == 0 || kernel_w == 0) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        state <= START_KERNEL_AREA;
                    end
                end

                START_KERNEL_AREA: begin
                    mul_lhs_r <= {24'd0, kh_size_r}; mul_rhs_r <= {8'd0, kw_size_r};
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_KERNEL_AREA;
                end
                WAIT_KERNEL_AREA: if (mul_done) begin kernel_area_r <= mul_product[15:0]; state <= DECODE_M; end

                DECODE_M: begin
                    if (m_remaining_r >= out_w_r) begin
                        m_remaining_r <= m_remaining_r - out_w_r;
                        oh_r <= oh_r + 1'b1;
                    end else begin
                        ow_r <= m_remaining_r;
                        state <= DECODE_K_IC;
                    end
                end

                DECODE_K_IC: begin
                    if (k_remaining_r >= kernel_area_r) begin
                        k_remaining_r <= k_remaining_r - kernel_area_r;
                        base_ic_r <= base_ic_r + 1'b1;
                    end else begin
                        state <= DECODE_K_KH;
                    end
                end

                DECODE_K_KH: begin
                    if (k_remaining_r >= kw_size_r) begin
                        k_remaining_r <= k_remaining_r - kw_size_r;
                        base_kh_r <= base_kh_r + 1'b1;
                    end else begin
                        base_kw_r <= k_remaining_r;
                        ic_r <= base_ic_r; kh_r <= base_kh_r; kw_r <= k_remaining_r;
                        state <= START_OH;
                    end
                end

                START_OH: begin
                    mul_lhs_r <= {16'd0, oh_r}; mul_rhs_r <= {8'd0, sh_r};
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_OH;
                end
                WAIT_OH: if (mul_done) begin
                    spatial_ih_r <= $signed(mul_product[16:0]) - $signed({1'b0, ph_r});
                    state <= START_OW;
                end
                START_OW: begin
                    mul_lhs_r <= {16'd0, ow_r}; mul_rhs_r <= {8'd0, sw_r};
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_OW;
                end
                WAIT_OW: if (mul_done) begin
                    spatial_iw_r <= $signed(mul_product[16:0]) - $signed({1'b0, pw_r});
                    state <= START_CHANNEL_STRIDE;
                end
                START_CHANNEL_STRIDE: begin
                    mul_lhs_r <= {16'd0, in_h_r}; mul_rhs_r <= in_w_r;
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_CHANNEL_STRIDE;
                end
                WAIT_CHANNEL_STRIDE: if (mul_done) begin channel_stride_r <= mul_product[31:0]; state <= START_ROW_HSTEP; end
                START_ROW_HSTEP: begin
                    mul_lhs_r <= {24'd0, sh_r}; mul_rhs_r <= in_w_r;
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_ROW_HSTEP;
                end
                WAIT_ROW_HSTEP: if (mul_done) begin row_hstep_r <= mul_product[31:0]; state <= START_ROW_WSPAN; end
                START_ROW_WSPAN: begin
                    mul_lhs_r <= {16'd0, out_w_r - 1'b1}; mul_rhs_r <= {8'd0, sw_r};
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_ROW_WSPAN;
                end
                WAIT_ROW_WSPAN: if (mul_done) begin row_wspan_r <= mul_product[31:0]; state <= START_KH_SPAN; end
                START_KH_SPAN: begin
                    mul_lhs_r <= {24'd0, kh_size_r - 1'b1}; mul_rhs_r <= in_w_r;
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_KH_SPAN;
                end
                WAIT_KH_SPAN: if (mul_done) begin
                    channel_wrap_delta_r <= channel_stride_r - mul_product[31:0];
                    state <= START_K_SEED;
                end
                START_K_SEED: begin
                    mul_lhs_r <= {16'd0, base_ic_r}; mul_rhs_r <= in_h_r;
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_K_SEED;
                end
                WAIT_K_SEED: if (mul_done) begin k_seed_r <= mul_product[31:0]; state <= INIT_K_ADD; end
                INIT_K_ADD: begin k_add_r <= k_seed_r + base_kh_r; state <= START_K_OFFSET; end
                START_K_OFFSET: begin
                    mul_lhs_r <= k_add_r; mul_rhs_r <= in_w_r;
                    mul_signed_r <= 1'b0; mul_start_r <= 1'b1; state <= WAIT_K_OFFSET;
                end
                WAIT_K_OFFSET: if (mul_done) begin k_offset_r <= mul_product[31:0] + base_kw_r; state <= START_SPATIAL; end
                START_SPATIAL: begin
                    mul_lhs_r <= {{15{spatial_ih_r[16]}}, spatial_ih_r}; mul_rhs_r <= in_w_r;
                    mul_signed_r <= 1'b1; mul_start_r <= 1'b1; state <= WAIT_SPATIAL;
                end
                WAIT_SPATIAL: if (mul_done) begin spatial_product_r <= $signed(mul_product[32:0]); state <= INIT_ADDRESS; end
                INIT_ADDRESS: begin
                    address_r <= $signed({1'b0, base_r}) + spatial_product_r + spatial_iw_r + $signed({1'b0, k_offset_r});
                    row_start_address_r <= $signed({1'b0, base_r}) + spatial_product_r + spatial_iw_r + $signed({1'b0, k_offset_r});
                    state <= CHECK_DESCRIPTOR;
                end

                CHECK_DESCRIPTOR: begin
                    checked_pad_r <= !current_valid;
                    checked_tile_left_r <= tile_left;
                    checked_kernel_left_r <= kernel_left;
                    checked_input_left_r <= input_left;
                    checked_line_left_r <= line_left;
                    prepared_len_r <= min_run_len(
                        tile_left, kernel_left, input_left, line_left);
                    state <= RUN;
                end

                RUN: if (descriptor_valid && descriptor_ready) begin
                    accepted_len_r <= {1'b0, descriptor_len};
                    accepted_last_r <= descriptor_last;
                    // The prepared descriptor is already registered and
                    // stable in RUN. Advance its recurrence on acceptance
                    // instead of spending a separate dead ADVANCE cycle.
                    if (descriptor_last) begin
                        state <= FINISH;
                    end else if (col_r + descriptor_len >= cols_r) begin
                        row_r <= row_r + 1'b1;
                        col_r <= '0;
                        ic_r <= base_ic_r; kh_r <= base_kh_r; kw_r <= base_kw_r;
                        if (ow_r + 1'b1 >= out_w_r) begin
                            oh_r <= oh_r + 1'b1;
                            ow_r <= '0;
                            spatial_ih_r <= spatial_ih_r + $signed({1'b0, sh_r});
                            spatial_iw_r <= -$signed({1'b0, pw_r});
                            address_r <= row_start_address_r + $signed({1'b0, row_hstep_r}) - $signed({1'b0, row_wspan_r});
                            row_start_address_r <= row_start_address_r + $signed({1'b0, row_hstep_r}) - $signed({1'b0, row_wspan_r});
                        end else begin
                            ow_r <= ow_r + 1'b1;
                            spatial_iw_r <= spatial_iw_r + $signed({1'b0, sw_r});
                            address_r <= row_start_address_r + $signed({1'b0, sw_r});
                            row_start_address_r <= row_start_address_r + $signed({1'b0, sw_r});
                        end
                    end else begin
                        col_r <= col_r + descriptor_len;
                        if (kw_r + descriptor_len >= kw_size_r) begin
                            kw_r <= '0;
                            if (kh_r + 1'b1 >= kh_size_r) begin
                                kh_r <= '0;
                                ic_r <= ic_r + 1'b1;
                                address_r <= address_r + $signed({1'b0, channel_wrap_delta_r}) - $signed({1'b0, kw_r});
                            end else begin
                                kh_r <= kh_r + 1'b1;
                                address_r <= address_r + $signed({1'b0, in_w_r}) - $signed({1'b0, kw_r});
                            end
                        end else begin
                            kw_r <= kw_r + descriptor_len;
                            address_r <= address_r + $signed({1'b0, descriptor_len});
                        end
                    end
                    if (!descriptor_last) state <= CHECK_DESCRIPTOR;
                end

                FINISH: state <= IDLE;
                FAIL: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
