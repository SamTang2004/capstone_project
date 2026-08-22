// Direct DDR im2col tile loader. Coordinate decode is sequential: no / or %
// operators are present on the DDR request path. Contiguous activation kernel
// rows and weight K-runs stream from cache-line AXI bursts.
module conv2d_tile_loader_ddr #(
    parameter int TILE_SIZE = 16,
    parameter bit WIDE_DATA_ENABLE = 1'b0
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
    output logic [6:0] read_req_len,
    input logic read_data_valid, output logic read_data_ready, input logic [7:0] read_data,
    input logic read_data_last,
    input logic read_word_valid, output logic read_word_ready,
    input logic [63:0] read_word_data, input logic [7:0] read_word_keep,
    input logic read_word_last,
    output logic cache_we, output logic cache_matrix, output logic [7:0] cache_addr, output logic [7:0] cache_wdata,
    output logic cache_word_we, output logic [7:0] cache_word_addr,
    output logic [7:0] cache_word_stride, output logic [63:0] cache_word_data,
    output logic [7:0] cache_word_keep,
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
        ADDR_ADD2, ADDR_BASE, BURST_CALC, BURST_REDUCE, BURST_PREP,
        PRESLICE_TILE_MUL, PRESLICE_TILE_BASE,
        REQUEST, RECEIVE, ADVANCE, JIT_ACTIVE, FINISH, FAIL
    } state_t;
    state_t state;
    logic matrix_r, conv_r, presliced_r, pad_r;
    logic [4:0] row_r, col_r, rows_r, cols_r;
    logic [31:0] normal_row_base_r, normal_stride_r, request_addr_r;
    logic [15:0] in_c_r, in_h_r, in_w_r, out_c_r, out_w_r, n0_r;
    logic [7:0] kh_size_r, kw_size_r, sh_r, sw_r, ph_r, pw_r;
    logic [31:0] input_base_r, weight_base_r;
    logic [15:0] m_remaining, k_remaining, kernel_area_r;
    logic [15:0] row_oh_r, row_ow_r, base_oh_r, base_ow_r;
    logic [15:0] k_ic_r, k_kh_r, k_kw_r, base_ic_r, base_kh_r, base_kw_r;
    // Each address stage owns its output register. Reusing two temporaries
    // created a large state-controlled mux on addr_term0_r in routed logic.
    (* use_dsp = "yes" *) logic [31:0] preslice_tile_index_r;
    logic [31:0] preslice_tile_base_r;
    logic [8:0] preslice_linear_r;
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
    logic [6:0] burst_len_calc_r, run_len_r, run_remaining_r;
    logic [16:0] burst_pair0_r, burst_pair1_r;
    logic final_row_r, final_col_r;
    (* use_dsp = "yes" *) logic [31:0] coord_h_product_r, coord_w_product_r;

    logic jit_address_start, jit_address_busy, jit_address_done, jit_address_error;
    logic jit_descriptor_valid, jit_descriptor_ready, jit_descriptor_pad, jit_descriptor_last;
    logic [31:0] jit_descriptor_addr;
    logic [4:0] jit_descriptor_len, jit_descriptor_row, jit_descriptor_col;
    logic [31:0] jit_fifo_addr [0:3];
    logic [6:0] jit_fifo_len [0:3];
    logic [4:0] jit_fifo_row [0:3], jit_fifo_col [0:3];
    logic jit_fifo_pad [0:3], jit_fifo_last [0:3];
    logic [1:0] jit_fifo_write_ptr, jit_fifo_read_ptr;
    logic [2:0] jit_fifo_count;
    logic [31:0] jit_tail_addr;
    logic [31:0] jit_tail_next_addr;
    logic [6:0] jit_tail_len;
    logic [6:0] jit_tail_room;
    logic [5:0] jit_tail_next_col;
    logic [4:0] jit_tail_row, jit_tail_col;
    logic jit_tail_pad, jit_tail_last;
    typedef enum logic [1:0] {JIT_BACKEND_IDLE, JIT_BACKEND_REQUEST, JIT_BACKEND_RECEIVE} jit_backend_state_t;
    jit_backend_state_t jit_backend_state;
    logic [31:0] jit_active_addr;
    logic [6:0] jit_active_len, jit_active_remaining;
    logic [4:0] jit_active_row, jit_active_col;
    logic jit_active_pad, jit_active_last;

    logic [6:0] jit_response_remaining [0:7];
    logic [4:0] jit_response_row [0:7], jit_response_col [0:7];
    logic jit_response_pad [0:7], jit_response_last [0:7];
    logic [2:0] jit_response_write_ptr, jit_response_read_ptr;
    logic [3:0] jit_response_count;
    wire jit_backend_complete;

    wire [1:0] jit_fifo_tail_ptr = jit_fifo_write_ptr - 1'b1;
    wire jit_merge_candidate = state == JIT_ACTIVE && jit_descriptor_valid &&
        jit_fifo_count > 1 && !jit_descriptor_pad &&
        !jit_tail_pad && !jit_tail_last &&
        jit_tail_row == jit_descriptor_row &&
        jit_tail_next_col == {1'b0, jit_descriptor_col} &&
        jit_tail_next_addr == jit_descriptor_addr &&
        {2'd0, jit_descriptor_len} <= jit_tail_room &&
        {jit_tail_addr[31:6], 6'd0} ==
            {jit_descriptor_addr[31:6], 6'd0};
    wire jit_enqueue = state == JIT_ACTIVE && jit_descriptor_valid && jit_descriptor_ready;
    wire jit_merge = jit_enqueue && jit_merge_candidate;
    wire jit_enqueue_new = jit_enqueue && !jit_merge;
    wire jit_request_accept = state == JIT_ACTIVE &&
                              jit_backend_state == JIT_BACKEND_REQUEST &&
                              read_req_valid && read_req_ready;
    assign jit_backend_complete = jit_request_accept ||
                                  (state == JIT_ACTIVE &&
                                   jit_backend_state == JIT_BACKEND_RECEIVE);
    wire jit_response_enqueue = jit_backend_complete;
    // The memory driver guarantees contiguous qword lanes. Architectural
    // progress therefore follows the requested byte count, while keep is
    // forwarded solely as the final-beat write mask.
    wire [3:0] receive_word_bytes = run_remaining_r > 8
        ? 4'd8 : run_remaining_r[3:0];
    wire [3:0] jit_word_bytes = jit_response_remaining[jit_response_read_ptr] > 8
        ? 4'd8 : jit_response_remaining[jit_response_read_ptr][3:0];
    wire jit_response_word = state == JIT_ACTIVE && jit_response_count != 0 &&
                             !jit_response_pad[jit_response_read_ptr] &&
                             (WIDE_DATA_ENABLE ? read_word_valid && read_word_ready
                                               : read_data_valid && read_data_ready);
    wire jit_response_pop = state == JIT_ACTIVE && jit_response_count != 0 &&
                            ((jit_response_pad[jit_response_read_ptr] &&
                              jit_response_remaining[jit_response_read_ptr] == 1) ||
                             (jit_response_word &&
                              (WIDE_DATA_ENABLE
                                  ? jit_response_remaining[jit_response_read_ptr] <= 8
                                  : jit_response_remaining[jit_response_read_ptr] == 1)));
    // A dequeued descriptor owns a future response slot. When replacing a
    // descriptor that completes now, reserve space for both metadata writes;
    // otherwise a pad descriptor can overflow and corrupt the ordered ring.
    wire jit_dequeue = state == JIT_ACTIVE && jit_fifo_count != 0 &&
                       ((jit_backend_state == JIT_BACKEND_IDLE && jit_response_count < 8) ||
                        (jit_backend_complete &&
                         (jit_response_count < 7 ||
                          (jit_response_count == 7 && jit_response_pop))));

    conv2d_activation_address_gen #(.TILE_SIZE(TILE_SIZE)) activation_address_gen (
        .clk(clk), .rst_n(rst_n), .start(jit_address_start),
        .input_base_addr(input_base_addr), .input_c(input_c), .input_h(input_h), .input_w(input_w),
        .out_w(out_w), .kernel_h(kernel_h), .kernel_w(kernel_w),
        .stride_h(stride_h), .stride_w(stride_w), .pad_h(pad_h), .pad_w(pad_w),
        .m0(m0), .k0(k0), .tile_rows(tile_rows), .tile_cols(tile_cols),
        .busy(jit_address_busy), .done(jit_address_done), .error(jit_address_error),
        .descriptor_valid(jit_descriptor_valid), .descriptor_ready(jit_descriptor_ready),
        .descriptor_addr(jit_descriptor_addr), .descriptor_len(jit_descriptor_len),
        .descriptor_pad(jit_descriptor_pad), .descriptor_row(jit_descriptor_row),
        .descriptor_col(jit_descriptor_col), .descriptor_last(jit_descriptor_last)
    );

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign jit_descriptor_ready = state == JIT_ACTIVE &&
                                  (jit_fifo_count < 4 || jit_merge_candidate);
    assign read_req_valid = state == JIT_ACTIVE
        ? jit_backend_state == JIT_BACKEND_REQUEST && jit_response_count < 8
        : state == REQUEST;
    assign read_req_addr = state == JIT_ACTIVE ? jit_active_addr : request_addr_r;
    assign read_req_len = state == JIT_ACTIVE ? jit_active_len : run_len_r;
    assign read_data_ready = !WIDE_DATA_ENABLE && (state == JIT_ACTIVE
        ? jit_response_count != 0 && !jit_response_pad[jit_response_read_ptr]
        : state == RECEIVE && !pad_r);
    assign read_word_ready = WIDE_DATA_ENABLE && (state == JIT_ACTIVE
        ? jit_response_count != 0 && !jit_response_pad[jit_response_read_ptr]
        : state == RECEIVE && !pad_r);
    wire cache_we_next = state == JIT_ACTIVE
        ? jit_response_count != 0 &&
          (jit_response_pad[jit_response_read_ptr] || (!WIDE_DATA_ENABLE && read_data_valid && read_data_ready))
        : state == RECEIVE && (pad_r || (!WIDE_DATA_ENABLE && read_data_valid));
    wire [7:0] cache_addr_next = state == JIT_ACTIVE
        ? jit_response_row[jit_response_read_ptr] * TILE_SIZE + jit_response_col[jit_response_read_ptr]
        : matrix_r && presliced_r ? preslice_linear_r[7:0] : row_r * TILE_SIZE + col_r;
    wire [7:0] cache_wdata_next =
        (state == JIT_ACTIVE ? jit_response_pad[jit_response_read_ptr] : pad_r) ? 8'h00 : read_data;
    wire cache_word_we_next = WIDE_DATA_ENABLE && read_word_valid && read_word_ready;
    wire [7:0] cache_word_addr_next = state == JIT_ACTIVE
        ? jit_response_row[jit_response_read_ptr] * TILE_SIZE + jit_response_col[jit_response_read_ptr]
        : matrix_r && presliced_r ? preslice_linear_r[7:0] : row_r * TILE_SIZE + col_r;
    wire [7:0] cache_word_stride_next = state == JIT_ACTIVE || !matrix_r || presliced_r ? 8'd1 : TILE_SIZE;
    assign profile_request_phase = state == JIT_ACTIVE ? jit_backend_state == JIT_BACKEND_REQUEST : state == REQUEST;
    assign profile_receive_phase = state == JIT_ACTIVE ? jit_response_count != 0 : state == RECEIVE;
    assign profile_address_phase = state == JIT_ACTIVE ? jit_address_busy : busy && state != REQUEST && state != RECEIVE;

    wire [16:0] burst_tile_left = (matrix_r && !presliced_r) ? rows_r - row_r : cols_r - col_r;
    wire [16:0] burst_kernel_left = matrix_r ? TILE_SIZE : kw_size_r - k_kw_r;
    wire [16:0] burst_input_left = matrix_r ? TILE_SIZE : in_w_r - addr_iw_r;
    wire [16:0] burst_line_left = 17'd64 - {11'd0, request_addr_r[5:0]};
    wire [16:0] burst_tile_cap = burst_tile_left > TILE_SIZE ? TILE_SIZE : burst_tile_left;
    wire [16:0] burst_pair0 = burst_tile_cap < burst_kernel_left ? burst_tile_cap : burst_kernel_left;
    wire [16:0] burst_pair1 = burst_input_left < burst_line_left ? burst_input_left : burst_line_left;
    wire [16:0] burst_min = burst_pair0_r < burst_pair1_r ? burst_pair0_r : burst_pair1_r;
    wire [31:0] preslice_next_addr = preslice_tile_base_r + preslice_linear_r;
    wire [6:0] preslice_line_left = 7'd64 - {1'b0, preslice_next_addr[5:0]};
    wire [8:0] preslice_tile_left = 9'd256 - preslice_linear_r;
    wire [6:0] preslice_run_len = preslice_tile_left < preslice_line_left
        ? preslice_tile_left[6:0] : preslice_line_left;

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
            cache_we <= 1'b0; cache_matrix <= 1'b0; cache_addr <= '0; cache_wdata <= '0;
            cache_word_we <= 1'b0; cache_word_addr <= '0; cache_word_stride <= '0;
            cache_word_data <= '0; cache_word_keep <= '0;
            rows_r <= 0; cols_r <= 0; request_addr_r <= 0; m_remaining <= 0; k_remaining <= 0;
            row_oh_r <= 0; row_ow_r <= 0; base_oh_r <= 0; base_ow_r <= 0; k_ic_r <= 0; k_kh_r <= 0; k_kw_r <= 0; base_ic_r <= 0; base_kh_r <= 0; base_kw_r <= 0;
            normal_row_base_r <= 0; preslice_tile_index_r <= 0; preslice_tile_base_r <= 0; preslice_linear_r <= 0; seed_product_r <= 0; addr_add0_r <= 0; addr_mul1_r <= 0; addr_add1_r <= 0; addr_mul2_r <= 0; addr_add2_r <= 0;
            addr_mul1_rhs_r <= 0; addr_mul1_chunk0_r <= 0; addr_mul1_chunk1_r <= 0; addr_mul1_pair_low_r <= 0; addr_mul1_pair_high_r <= 0;
            addr_seed_lhs_r <= 0; addr_seed_rhs_r <= 0; addr_ih_r <= 0; addr_iw_r <= 0; addr_weight_r <= 0; addr_oc_r <= 0; coord_h_product_r <= 0; coord_w_product_r <= 0;
            burst_len_calc_r <= 7'd1; run_len_r <= 7'd1; run_remaining_r <= 7'd1; final_row_r <= 1'b0; final_col_r <= 1'b0;
            burst_pair0_r <= '0; burst_pair1_r <= '0;
            jit_fifo_write_ptr <= '0; jit_fifo_read_ptr <= '0; jit_fifo_count <= '0;
            jit_tail_addr <= '0; jit_tail_next_addr <= '0; jit_tail_len <= '0; jit_tail_room <= '0;
            jit_tail_next_col <= '0; jit_tail_row <= '0; jit_tail_col <= '0;
            jit_tail_pad <= 1'b0; jit_tail_last <= 1'b0;
            jit_address_start <= 1'b0;
            jit_backend_state <= JIT_BACKEND_IDLE;
            jit_active_addr <= '0; jit_active_len <= '0; jit_active_remaining <= '0;
            jit_active_row <= '0; jit_active_col <= '0; jit_active_pad <= 1'b0; jit_active_last <= 1'b0;
            jit_response_write_ptr <= '0; jit_response_read_ptr <= '0; jit_response_count <= '0;
        end else begin
            done <= 0;
            jit_address_start <= 1'b0;
            cache_we <= cache_we_next;
            cache_matrix <= matrix_r;
            cache_addr <= cache_addr_next;
            cache_wdata <= cache_wdata_next;
            cache_word_we <= cache_word_we_next;
            cache_word_addr <= cache_word_addr_next;
            cache_word_stride <= cache_word_stride_next;
            cache_word_data <= read_word_data;
            cache_word_keep <= read_word_keep;
            case (state)
                IDLE: if (start) begin
                    matrix_r<=matrix_select; conv_r<=conv_mode; presliced_r<=presliced_weights; row_r<=0; col_r<=0; rows_r<=tile_rows; cols_r<=tile_cols; error<=0;
                    normal_row_base_r<=base_addr; normal_stride_r<=row_stride; input_base_r<=input_base_addr; weight_base_r<=weight_base_addr;
                    in_c_r<=input_c; in_h_r<=input_h; in_w_r<=input_w; out_c_r<=out_c; out_w_r<=out_w; n0_r<=n0;
                    kh_size_r<=kernel_h; kw_size_r<=kernel_w; sh_r<=stride_h; sw_r<=stride_w; ph_r<=pad_h; pw_r<=pad_w; kernel_area_r<=kernel_h*kernel_w;
                    m_remaining<=m0; k_remaining<=k0;
                    base_oh_r<=0; base_ow_r<=0; row_oh_r<=0; row_ow_r<=0;
                    base_ic_r<=0; base_kh_r<=0; base_kw_r<=0; k_ic_r<=0; k_kh_r<=0; k_kw_r<=0;
                    preslice_linear_r <= '0;
                    if (tile_rows==0 || tile_rows>TILE_SIZE || tile_cols==0 || tile_cols>TILE_SIZE || (conv_mode && (out_w==0 || kernel_h==0 || kernel_w==0))) begin error<=1; state<=FAIL; end
                    else if (conv_mode && !matrix_select) begin
                        jit_fifo_write_ptr <= '0; jit_fifo_read_ptr <= '0; jit_fifo_count <= '0;
                        jit_response_write_ptr <= '0; jit_response_read_ptr <= '0; jit_response_count <= '0;
                        jit_backend_state <= JIT_BACKEND_IDLE;
                        jit_address_start <= 1'b1;
                        state <= JIT_ACTIVE;
                    end else if (!conv_mode) state<=BUILD;
                    else if (matrix_select && presliced_weights) state<=PRESLICE_TILE_MUL;
                    else state<=DECODE_M;
                end

                JIT_ACTIVE: begin
                    if (jit_address_done && jit_address_error) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        if (jit_merge) begin
                            jit_fifo_len[jit_fifo_tail_ptr] <=
                                jit_tail_len + jit_descriptor_len;
                            jit_fifo_last[jit_fifo_tail_ptr] <= jit_descriptor_last;
                            jit_tail_len <= jit_tail_len + jit_descriptor_len;
                            jit_tail_next_addr <= jit_tail_next_addr + jit_descriptor_len;
                            jit_tail_next_col <= jit_tail_next_col + jit_descriptor_len;
                            jit_tail_room <= jit_tail_room - jit_descriptor_len;
                            jit_tail_last <= jit_descriptor_last;
                        end else if (jit_enqueue_new) begin
                            jit_fifo_addr[jit_fifo_write_ptr] <= jit_descriptor_addr;
                            jit_fifo_len[jit_fifo_write_ptr] <= {2'd0, jit_descriptor_len};
                            jit_fifo_row[jit_fifo_write_ptr] <= jit_descriptor_row;
                            jit_fifo_col[jit_fifo_write_ptr] <= jit_descriptor_col;
                            jit_fifo_pad[jit_fifo_write_ptr] <= jit_descriptor_pad;
                            jit_fifo_last[jit_fifo_write_ptr] <= jit_descriptor_last;
                            jit_fifo_write_ptr <= jit_fifo_write_ptr + 1'b1;
                            jit_tail_addr <= jit_descriptor_addr;
                            jit_tail_next_addr <= jit_descriptor_addr + jit_descriptor_len;
                            jit_tail_len <= {2'd0, jit_descriptor_len};
                            jit_tail_room <= 7'd64 - jit_descriptor_len;
                            jit_tail_next_col <= {1'b0, jit_descriptor_col} + jit_descriptor_len;
                            jit_tail_row <= jit_descriptor_row;
                            jit_tail_col <= jit_descriptor_col;
                            jit_tail_pad <= jit_descriptor_pad;
                            jit_tail_last <= jit_descriptor_last;
                        end
                        if (jit_dequeue) begin
                            jit_active_addr <= jit_fifo_addr[jit_fifo_read_ptr];
                            jit_active_len <= jit_fifo_len[jit_fifo_read_ptr];
                            jit_active_remaining <= jit_fifo_len[jit_fifo_read_ptr];
                            jit_active_row <= jit_fifo_row[jit_fifo_read_ptr];
                            jit_active_col <= jit_fifo_col[jit_fifo_read_ptr];
                            jit_active_pad <= jit_fifo_pad[jit_fifo_read_ptr];
                            jit_active_last <= jit_fifo_last[jit_fifo_read_ptr];
                            jit_fifo_read_ptr <= jit_fifo_read_ptr + 1'b1;
                            jit_backend_state <= jit_fifo_pad[jit_fifo_read_ptr]
                                ? JIT_BACKEND_RECEIVE : JIT_BACKEND_REQUEST;
                        end
                        case ({jit_enqueue_new, jit_dequeue})
                            2'b10: jit_fifo_count <= jit_fifo_count + 1'b1;
                            2'b01: jit_fifo_count <= jit_fifo_count - 1'b1;
                            default: begin end
                        endcase

                        if (jit_response_enqueue) begin
                            jit_response_remaining[jit_response_write_ptr] <= jit_active_len;
                            jit_response_row[jit_response_write_ptr] <= jit_active_row;
                            jit_response_col[jit_response_write_ptr] <= jit_active_col;
                            jit_response_pad[jit_response_write_ptr] <= jit_active_pad;
                            jit_response_last[jit_response_write_ptr] <= jit_active_last;
                            jit_response_write_ptr <= jit_response_write_ptr + 1'b1;
                        end

                        if (jit_response_count != 0) begin
                            if (jit_response_pad[jit_response_read_ptr]) begin
                                if (jit_response_remaining[jit_response_read_ptr] > 1) begin
                                    jit_response_remaining[jit_response_read_ptr] <=
                                        jit_response_remaining[jit_response_read_ptr] - 1'b1;
                                    jit_response_col[jit_response_read_ptr] <=
                                        jit_response_col[jit_response_read_ptr] + 1'b1;
                                end
                            end else if (jit_response_word) begin
                                if (WIDE_DATA_ENABLE &&
                                    (read_word_last !=
                                         (jit_response_remaining[jit_response_read_ptr] <= 8))) begin
                                    error <= 1'b1;
                                    state <= FAIL;
                                end else if (!WIDE_DATA_ENABLE &&
                                             read_data_last !=
                                                 (jit_response_remaining[jit_response_read_ptr] == 1)) begin
                                    error <= 1'b1;
                                    state <= FAIL;
                                end else if (WIDE_DATA_ENABLE &&
                                             jit_response_remaining[jit_response_read_ptr] > jit_word_bytes) begin
                                    jit_response_remaining[jit_response_read_ptr] <=
                                        jit_response_remaining[jit_response_read_ptr] - jit_word_bytes;
                                    jit_response_col[jit_response_read_ptr] <=
                                        jit_response_col[jit_response_read_ptr] + jit_word_bytes;
                                end else if (!WIDE_DATA_ENABLE &&
                                             jit_response_remaining[jit_response_read_ptr] > 1) begin
                                    jit_response_remaining[jit_response_read_ptr] <=
                                        jit_response_remaining[jit_response_read_ptr] - 1'b1;
                                    jit_response_col[jit_response_read_ptr] <=
                                        jit_response_col[jit_response_read_ptr] + 1'b1;
                                end
                            end
                        end

                        if (jit_response_pop) begin
                            jit_response_read_ptr <= jit_response_read_ptr + 1'b1;
                            if (jit_response_last[jit_response_read_ptr]) state <= FINISH;
                        end
                        case ({jit_response_enqueue, jit_response_pop})
                            2'b10: jit_response_count <= jit_response_count + 1'b1;
                            2'b01: jit_response_count <= jit_response_count - 1'b1;
                            default: begin end
                        endcase

                        case (jit_backend_state)
                            JIT_BACKEND_REQUEST: begin
                                if (jit_request_accept && !jit_dequeue)
                                    jit_backend_state <= JIT_BACKEND_IDLE;
                            end
                            JIT_BACKEND_RECEIVE: begin
                                if (!jit_dequeue) jit_backend_state <= JIT_BACKEND_IDLE;
                            end
                            default: begin end
                        endcase
                    end
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
                        state <= NORMAL_FINAL;
                    end else if (!matrix_r) begin
                        addr_weight_r <= 1'b0;
                        state <= COORD_MUL;
                    end else if (presliced_r) begin
                        request_addr_r <= preslice_next_addr;
                        run_len_r <= preslice_run_len;
                        run_remaining_r <= preslice_run_len;
                        state <= preslice_run_len == 0 ? FAIL : REQUEST;
                    end else begin
                        addr_weight_r <= 1'b1;
                        addr_oc_r <= n0_r + col_r;
                        addr_ih_r <= $signed({1'b0, k_kh_r});
                        addr_iw_r <= $signed({1'b0, k_kw_r});
                        state <= ADDR_SEED;
                    end
                end
                NORMAL_FINAL: begin
                    request_addr_r <= normal_row_base_r + col_r;
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
                    // Register balanced pairwise limits before the final
                    // minimum so setup does not form a serial compare chain.
                    burst_pair0_r <= burst_pair0;
                    burst_pair1_r <= burst_pair1;
                    state <= BURST_REDUCE;
                end
                BURST_REDUCE: begin
                    burst_len_calc_r <= {2'd0, burst_min[4:0]};
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
                    if (pad_r || (!WIDE_DATA_ENABLE && read_data_valid) ||
                        (WIDE_DATA_ENABLE && read_word_valid && read_word_ready)) begin
                        // Stream all interior burst beats without bubbles. On
                        // the final beat, register boundary comparisons so the
                        // coordinate counters cannot feed the FSM state cone.
                        if (!pad_r && WIDE_DATA_ENABLE &&
                            read_word_last != (run_remaining_r <= 8)) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else if (!pad_r && !WIDE_DATA_ENABLE &&
                                     (read_data_last != (run_remaining_r == 1))) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else if (WIDE_DATA_ENABLE && !pad_r && run_remaining_r > receive_word_bytes) begin
                            run_remaining_r <= run_remaining_r - receive_word_bytes;
                            if (matrix_r && presliced_r) preslice_linear_r <= preslice_linear_r + receive_word_bytes;
                            else if (matrix_r) begin
                                row_r <= row_r + receive_word_bytes;
                                k_kw_r <= k_kw_r + receive_word_bytes;
                            end else col_r <= col_r + receive_word_bytes;
                        end else if (!WIDE_DATA_ENABLE && conv_r && run_remaining_r > 1) begin
                            run_remaining_r <= run_remaining_r - 1'b1;
                            if (matrix_r && presliced_r) preslice_linear_r <= preslice_linear_r + 1'b1;
                            else increment_k();
                            if (matrix_r && !presliced_r) row_r <= row_r + 1'b1;
                            else if (!(matrix_r && presliced_r)) col_r <= col_r + 1'b1;
                        end else begin
                            if (WIDE_DATA_ENABLE && !pad_r && receive_word_bytes > 1) begin
                                if (matrix_r && presliced_r) preslice_linear_r <= preslice_linear_r + receive_word_bytes;
                                else if (matrix_r) begin
                                    row_r <= row_r + receive_word_bytes - 1'b1;
                                    k_kw_r <= k_kw_r + receive_word_bytes - 1'b1;
                                end else col_r <= col_r + receive_word_bytes - 1'b1;
                            end
                            if (!WIDE_DATA_ENABLE && matrix_r && presliced_r)
                                preslice_linear_r <= preslice_linear_r + 1'b1;
                            final_row_r <= matrix_r && presliced_r ? 1'b1 :
                                WIDE_DATA_ENABLE && !pad_r && matrix_r && !presliced_r
                                ? row_r + receive_word_bytes >= rows_r
                                : row_r == rows_r - 1'b1;
                            final_col_r <= matrix_r && presliced_r
                                ? preslice_linear_r + (WIDE_DATA_ENABLE ? receive_word_bytes : 1'b1) >= 9'd256 :
                                WIDE_DATA_ENABLE && !pad_r && !(matrix_r && !presliced_r)
                                ? col_r + receive_word_bytes >= cols_r
                                : col_r == cols_r - 1'b1;
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
                            normal_row_base_r <= normal_row_base_r + normal_stride_r;
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
