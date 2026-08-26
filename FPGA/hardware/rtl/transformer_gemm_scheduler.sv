module transformer_gemm_scheduler #(
    parameter int MAX_K = 1024,
    parameter int MAX_N = 768,
    parameter int MAX_W4_CACHE_VECTORS = 18432,
    parameter int K_WIDTH = $clog2(MAX_K + 1)
) (
    input  logic clk, rst_n, pipeline_enable,
    input  logic start, weight_int4, fused_requant, fused_gelu,
    input logic [1:0] fused_output_mode,
    input  logic [31:0] a_base, b_base, c_base, c_stride,
    input  logic [31:0] bias_base,
    input  logic signed [31:0] rq_multiplier,
    input  logic [5:0] rq_right_shift,
    input  logic signed [31:0] rq_output_zero_point,
    input  logic [15:0] dim_m, dim_n, dim_k,
    output logic busy, done, error,
    output logic op_start, op_done,
    output logic memory_stall, input_stall, output_stall,
    output logic [31:0] debug_state,
    output logic [63:0] phase_load_a_cycles, phase_load_b_cycles,
    output logic [63:0] phase_load_b_weight_cycles, phase_load_b_bias_cycles,
    output logic [63:0] w4_cache_hit_tiles, w4_cache_miss_tiles,
    output logic [63:0] phase_compute_cycles, phase_write_cycles,

    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [6:0] read_req_len,
    output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input logic read_qword_valid, output logic read_qword_ready,
    input logic [63:0] read_qword_data, input logic [7:0] read_qword_keep,
    input logic read_qword_last,

    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last,
    input logic memory_busy, memory_error
);
    typedef enum logic [4:0] {
        IDLE, LOAD_A_PREP, LOAD_A_REQ, LOAD_A_RECV,
        LOAD_B_PREP, LOAD_B_REQ, LOAD_B_RECV,
        LOAD_BIAS_PREP, LOAD_BIAS_REQ, LOAD_BIAS_RECV,
        LOAD_BIAS_CACHE_PREFETCH, LOAD_BIAS_CACHE_RECV,
        CORE_START, CORE_PREFETCH, CORE_FEED, CORE_RESULT, WRITE_RESULT,
        FUSED_START, FUSED_WAIT, FUSED_DRAIN, WRITE_ARM, WRITE_WAIT,
        NEXT_TILE, FINISH, FAIL
    } state_t;
    state_t state;

    (* ram_style = "block" *) logic [63:0] activation_ram0 [0:MAX_K-1];
    (* ram_style = "block" *) logic [63:0] activation_ram1 [0:MAX_K-1];
    (* ram_style = "block" *) logic [63:0] weight_ram [0:MAX_K-1];
    // A complete DeiT-Tiny packed-W4 matrix is at most 147,456 elements, or
    // 18,432 32-bit K-vectors. Compact tile-major addressing uses the current
    // layer K as its stride instead of reserving MAX_K for every N tile.
    // Once fetched for the first M pair, later M pairs issue no B DDR read.
    localparam int W4_CACHE_TILES = (MAX_N + 7) / 8;
    // Store two packed 8-weight vectors per BRAM word.  DDR returns one
    // qword per beat, so a 64-byte cache-line request now fills sixteen
    // vectors without issuing sixteen independent 4-byte transactions.
    localparam int MAX_W4_CACHE_QWORDS = (MAX_W4_CACHE_VECTORS + 1) / 2;
    localparam int W4_CACHE_QW_AW = $clog2(MAX_W4_CACHE_QWORDS);
    (* ram_style = "block" *) logic [63:0]
        w4_weight_cache [0:MAX_W4_CACHE_QWORDS-1];
    logic [W4_CACHE_TILES-1:0] w4_cache_valid;
    logic [W4_CACHE_TILES-1:0] bias_cache_valid;
    // Four whole 64-bit words per N tile infer a compact synchronous BRAM.
    (* ram_style = "block" *) logic [63:0] bias_cache [0:W4_CACHE_TILES*4-1];
    logic [63:0] bias_cache_qword;
    logic [W4_CACHE_QW_AW-1:0] w4_cache_tile_base;
    logic [63:0] w4_cache_qword;
    logic w4_cache_half_q;
    logic [63:0] weight_ram_q;
    logic [63:0] feed_activation0, feed_activation1, feed_weight;
    logic [K_WIDTH-1:0] load_index, feed_index, b_issue_index;
    logic prefetch_active;
    logic [K_WIDTH-1:0] prefetch_issue_index, prefetch_recv_index;
    logic [31:0] prefetch_b_addr;
    logic [W4_CACHE_QW_AW-1:0] prefetch_cache_tile_base;
    logic [$clog2(W4_CACHE_TILES)-1:0] prefetch_cache_tile_index;
    logic [15:0] m_index, n_index;
    logic [3:0] valid_rows0, valid_rows1, valid_cols, write_row, write_col;
    logic load_a_bank, write_core_select, second_core_active;
    logic weight_int4_r;
    logic [31:0] a_base_r, b_base_r, c_base_r, c_stride_r;
    logic [31:0] bias_base_r;
    logic fused_requant_r, fused_gelu_r; logic [1:0] fused_output_mode_r;
    logic signed [31:0] rq_multiplier_r, rq_output_zero_point_r;
    logic [5:0] rq_right_shift_r;
    logic [255:0] bias_values;
    logic [31:0] a_tile_addr, b_tile_addr, c_tile_addr;
    logic [31:0] a_tile_stride, b_tile_stride, c_m_pair_stride, write_addr_r;
    logic [31:0] active_read_addr;
    logic [16:0] active_read_bytes_remaining;
    logic [6:0] active_read_line_remaining;
    logic [31:0] read_req_addr_r;
    logic [6:0] read_req_len_r;
    logic [15:0] m_r, n_r, k_r;
    logic core0_start, core0_busy, core0_input_valid, core0_input_ready;
    logic core0_result_valid, core0_result_ready, core0_error;
    logic core1_start, core1_busy, core1_input_valid, core1_input_ready;
    logic core1_result_valid, core1_result_ready, core1_error;
    logic [2047:0] accumulators0, accumulators1;
    logic [31:0] weight_bytes;
    logic activation0_write_en, activation1_write_en, weight_write_en, feed_advance;
    logic fused_writer_start, fused_writer_busy, fused_writer_done;
    logic fused_writer_capture_done, fused_writer_error;
    logic fused_write_req_valid, fused_write_req_ready, fused_write_req_last;
    logic [31:0] fused_write_req_addr, fused_write_req_data;
    logic [3:0] fused_write_req_strb;
    logic [K_WIDTH-1:0] ram_read_index;
    wire [W4_CACHE_QW_AW-1:0] w4_cache_write_index =
        w4_cache_tile_base + (load_index >> 1);
    wire [W4_CACHE_QW_AW-1:0] w4_cache_read_index =
        w4_cache_tile_base + (ram_read_index >> 1);
    wire [$clog2(W4_CACHE_TILES)-1:0] w4_cache_tile_index = n_index[15:3];
    wire [$clog2(W4_CACHE_TILES*4)-1:0] bias_cache_word_index =
        {w4_cache_tile_index, 2'b00} + load_index[1:0];
    wire [31:0] b_issue_addr = b_tile_addr +
        (weight_int4_r ? (b_issue_index << 2) : (b_issue_index << 3));
    wire [16:0] b_issue_bytes_remaining = (k_r - b_issue_index) *
        (weight_int4_r ? 17'd4 : 17'd8);
    wire [6:0] b_issue_line_remaining = 7'd64 - {1'b0, b_issue_addr[5:0]};
    wire [6:0] b_issue_len = b_issue_bytes_remaining < b_issue_line_remaining
        ? b_issue_bytes_remaining[6:0] : b_issue_line_remaining;
    wire [31:0] prefetch_issue_addr = prefetch_b_addr + (prefetch_issue_index << 2);
    wire [16:0] prefetch_bytes_remaining = (k_r - prefetch_issue_index) * 17'd4;
    wire [6:0] prefetch_line_remaining = 7'd64 - {1'b0, prefetch_issue_addr[5:0]};
    wire [6:0] prefetch_issue_len = prefetch_bytes_remaining < prefetch_line_remaining
        ? prefetch_bytes_remaining[6:0] : prefetch_line_remaining;
    wire foreground_read_req_valid =
        state == LOAD_A_REQ || state == LOAD_B_REQ || state == LOAD_BIAS_REQ ||
        (state == LOAD_B_RECV && b_issue_index < k_r);
    wire foreground_qword_ready =
        state == LOAD_A_RECV || state == LOAD_B_RECV || state == LOAD_BIAS_RECV;
    wire prefetch_bus_enable = prefetch_active && !foreground_read_req_valid &&
        !foreground_qword_ready;
    wire prefetch_read_req_valid = prefetch_bus_enable && prefetch_issue_index < k_r;
    wire prefetch_response_error = prefetch_bus_enable && read_qword_valid &&
        read_qword_ready &&
        (read_qword_keep != (prefetch_recv_index + 1'b1 == k_r ? 8'h0f : 8'hff) ||
         (prefetch_recv_index + (read_qword_keep[7:4] != 0 ? 2 : 1) == k_r &&
          !read_qword_last));
    wire [W4_CACHE_QW_AW-1:0] active_w4_cache_write_index = prefetch_bus_enable
        ? prefetch_cache_tile_base + (prefetch_recv_index >> 1)
        : w4_cache_write_index;

    assign weight_bytes = weight_int4_r ? 32'd4 : 32'd8;
    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;
    assign core0_start = state == CORE_START && pipeline_enable;
    assign core1_start = core0_start && second_core_active;
    /* Lock-step issue broadcasts B while each core consumes its own A tile.
     * Either core may stall without allowing the other to advance alone. */
    assign core0_input_valid = state == CORE_FEED && pipeline_enable &&
        (!second_core_active || core1_input_ready);
    assign core1_input_valid = state == CORE_FEED && pipeline_enable &&
        second_core_active && core0_input_ready;
    assign core0_result_ready = pipeline_enable &&
        ((!fused_requant_r && state == NEXT_TILE) ||
         (fused_requant_r && fused_writer_capture_done && !write_core_select));
    assign core1_result_ready = pipeline_enable && second_core_active &&
        ((!fused_requant_r && state == NEXT_TILE) ||
         (fused_requant_r && fused_writer_capture_done && write_core_select));
    assign feed_advance = core0_input_valid && core0_input_ready &&
        (!second_core_active || (core1_input_valid && core1_input_ready));
    assign feed_weight = weight_int4_r
        ? {32'd0, (w4_cache_half_q ? w4_cache_qword[63:32]
                                   : w4_cache_qword[31:0])}
        : weight_ram_q;

    // W4 B loading is a pipelined ordered stream.  While responses from the
    // current cache line are being consumed, queue following lines in the
    // memory driver's four-entry request FIFO.
    assign read_req_valid = pipeline_enable &&
        (foreground_read_req_valid || prefetch_read_req_valid);
    assign active_read_addr = state == LOAD_A_PREP
        ? a_tile_addr + (load_a_bank ? a_tile_stride : 0) + (load_index << 3)
        : state == LOAD_BIAS_PREP
            ? bias_base_r + (n_index << 2) + (load_index << 3)
            : b_tile_addr + (weight_int4_r ? (load_index << 2) : (load_index << 3));
    assign active_read_bytes_remaining = (k_r - load_index) *
        ((state == LOAD_B_PREP && weight_int4_r) ? 17'd4 : 17'd8);
    assign active_read_line_remaining = 7'd64 - {1'b0, active_read_addr[5:0]};
    assign read_req_addr = prefetch_read_req_valid ? prefetch_issue_addr :
        (state == LOAD_B_RECV ? b_issue_addr : read_req_addr_r);
    assign read_req_len = prefetch_read_req_valid ? prefetch_issue_len :
        (state == LOAD_B_RECV ? b_issue_len : read_req_len_r);
    wire [6:0] prepared_read_len = state == LOAD_BIAS_PREP
        ? ((load_index << 1) + 1 < valid_cols ? 7'd8 : 7'd4)
        : active_read_bytes_remaining < active_read_line_remaining
            ? active_read_bytes_remaining[6:0] : active_read_line_remaining;
    assign read_req_wide = 1'b1;
    assign read_req_qword = 1'b1;
    assign read_req_allow_ahead = weight_int4_r &&
        (state == LOAD_B_REQ || state == LOAD_B_RECV || prefetch_bus_enable);
    assign read_qword_ready = pipeline_enable &&
        (foreground_qword_ready || prefetch_bus_enable);

    assign write_req_valid = fused_requant_r ? fused_write_req_valid :
        state == WRITE_RESULT && pipeline_enable;
    assign write_req_addr = fused_requant_r ? fused_write_req_addr : write_addr_r;
    assign write_req_data = fused_requant_r ? fused_write_req_data : (write_core_select
        ? accumulators1[(write_row*8+write_col)*32 +: 32]
        : accumulators0[(write_row*8+write_col)*32 +: 32]);
    assign write_req_strb = fused_requant_r ? fused_write_req_strb : 4'hf;
    assign write_req_last = fused_requant_r ? fused_write_req_last :
        write_col + 1'b1 == valid_cols;
    assign fused_write_req_ready = fused_requant_r && write_req_ready;

    assign memory_stall = busy && ((foreground_read_req_valid && !read_req_ready) ||
                                   (foreground_qword_ready && !read_qword_valid) ||
                                   (write_req_valid && !write_req_ready));
    assign input_stall = state == CORE_FEED &&
        (!core0_input_ready || (second_core_active && !core1_input_ready));
    assign output_stall = state == CORE_RESULT &&
        (!core0_result_valid || (second_core_active && !core1_result_valid));
    assign debug_state = {15'd0, second_core_active, core1_error, core0_error,
                          core1_result_valid, core0_result_valid,
                          core1_input_ready, core0_input_ready,
                          memory_busy, 4'd0, state};

    assign activation0_write_en = state == LOAD_A_RECV && read_qword_valid &&
        read_qword_ready && !load_a_bank;
    assign activation1_write_en = state == LOAD_A_RECV && read_qword_valid &&
        read_qword_ready && load_a_bank;
    assign weight_write_en = (state == LOAD_B_RECV || prefetch_bus_enable) &&
        read_qword_valid && read_qword_ready;
    assign ram_read_index = state == CORE_FEED && feed_advance &&
        feed_index + 1'b1 < k_r ? feed_index + 1'b1 : feed_index;

    // Canonical synchronous simple-dual-port memories. Keeping the three K
    // vector stores in BRAM avoids thousands of distributed-RAM LUTs and the
    // associated routing congestion once both 8x8 arrays are present.
    always_ff @(posedge clk) begin
        if (activation0_write_en) activation_ram0[load_index] <= read_qword_data;
        feed_activation0 <= activation_ram0[ram_read_index];
    end
    always_ff @(posedge clk) begin
        if (activation1_write_en) activation_ram1[load_index] <= read_qword_data;
        feed_activation1 <= activation_ram1[ram_read_index];
    end
    always_ff @(posedge clk) begin
        if (weight_write_en && !weight_int4_r)
            weight_ram[load_index] <= read_qword_data;
        weight_ram_q <= weight_ram[ram_read_index];
    end
    always_ff @(posedge clk) begin
        if (weight_write_en && weight_int4_r)
            w4_weight_cache[active_w4_cache_write_index] <= read_qword_data;
        w4_cache_qword <= w4_weight_cache[w4_cache_read_index];
        w4_cache_half_q <= ram_read_index[0];
    end
    always_ff @(posedge clk) begin
        if (state == LOAD_BIAS_RECV && read_qword_valid && read_qword_ready)
            bias_cache[bias_cache_word_index] <= read_qword_data;
        bias_cache_qword <= bias_cache[bias_cache_word_index];
    end

    gemm_8x8_tile_core core0 (
        .clk, .rst_n, .start(core0_start), .weight_int4(weight_int4_r),
        .k_steps(k_r), .valid_rows(valid_rows0), .valid_cols,
        .input_valid(core0_input_valid), .input_ready(core0_input_ready),
        .activation_vector(feed_activation0), .packed_weight_vector(feed_weight),
        .busy(core0_busy), .result_valid(core0_result_valid),
        .result_ready(core0_result_ready), .error(core0_error), .accumulators(accumulators0)
    );
    gemm_8x8_tile_core core1 (
        .clk, .rst_n, .start(core1_start), .weight_int4(weight_int4_r),
        .k_steps(k_r), .valid_rows(valid_rows1), .valid_cols,
        .input_valid(core1_input_valid), .input_ready(core1_input_ready),
        .activation_vector(feed_activation1), .packed_weight_vector(feed_weight),
        .busy(core1_busy), .result_valid(core1_result_valid),
        .result_ready(core1_result_ready), .error(core1_error), .accumulators(accumulators1)
    );

    assign fused_writer_start = state == FUSED_START && pipeline_enable;
    gemm_requantize_writer fused_writer (
        .clk, .rst_n, .pipeline_enable, .start(fused_writer_start),
        .gelu_enable(fused_gelu_r),
        .accumulators(write_core_select ? accumulators1 : accumulators0),
        .bias_values, .bias_enable(bias_base_r != 0),
        .valid_rows(write_core_select ? valid_rows1 : valid_rows0), .valid_cols,
        .output_mode(fused_output_mode_r), .output_row_stride(c_stride_r),
        .output_base_addr(write_core_select ? c_tile_addr +
            (fused_output_mode_r == 2'd1 ? (c_stride_r << 3) :
             fused_output_mode_r == 2'd2 ? 32'd64 : (n_r << 3)) : c_tile_addr),
        .multiplier(rq_multiplier_r), .right_shift(rq_right_shift_r),
        .output_zero_point(rq_output_zero_point_r),
        .busy(fused_writer_busy), .done(fused_writer_done),
        .capture_done(fused_writer_capture_done), .error(fused_writer_error),
        .write_req_valid(fused_write_req_valid), .write_req_ready(fused_write_req_ready),
        .write_req_addr(fused_write_req_addr), .write_req_data(fused_write_req_data),
        .write_req_strb(fused_write_req_strb), .write_req_last(fused_write_req_last)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            load_index <= '0; feed_index <= '0; b_issue_index <= '0; m_index <= '0; n_index <= '0;
            valid_rows0 <= '0; valid_rows1 <= '0; valid_cols <= '0;
            write_row <= '0; write_col <= '0; load_a_bank <= 1'b0;
            write_core_select <= 1'b0; second_core_active <= 1'b0;
            weight_int4_r <= 1'b0; fused_requant_r <= 1'b0; fused_gelu_r <= 1'b0;
            fused_output_mode_r <= 0;
            a_base_r <= '0; b_base_r <= '0; bias_base_r <= '0;
            c_base_r <= '0; c_stride_r <= '0; m_r <= '0; n_r <= '0; k_r <= '0;
            rq_multiplier_r <= '0; rq_right_shift_r <= '0;
            rq_output_zero_point_r <= '0; bias_values <= '0;
            a_tile_addr <= '0; b_tile_addr <= '0; c_tile_addr <= '0;
            a_tile_stride <= '0; b_tile_stride <= '0; c_m_pair_stride <= '0; write_addr_r <= '0;
            read_req_addr_r <= '0; read_req_len_r <= '0;
            w4_cache_valid <= '0; bias_cache_valid <= '0; w4_cache_tile_base <= '0;
            prefetch_active <= 1'b0; prefetch_issue_index <= '0; prefetch_recv_index <= '0;
            prefetch_b_addr <= '0; prefetch_cache_tile_base <= '0;
            prefetch_cache_tile_index <= '0;
            phase_load_a_cycles <= '0; phase_load_b_cycles <= '0;
            phase_load_b_weight_cycles <= '0; phase_load_b_bias_cycles <= '0;
            w4_cache_hit_tiles <= '0; w4_cache_miss_tiles <= '0;
            phase_compute_cycles <= '0; phase_write_cycles <= '0;
        end else begin
            done <= 1'b0;
            if (pipeline_enable) begin
                if (prefetch_read_req_valid && read_req_ready)
                    prefetch_issue_index <= prefetch_issue_index + (read_req_len >> 2);
                if (prefetch_bus_enable && read_qword_valid && read_qword_ready) begin
                    if (!prefetch_response_error && prefetch_recv_index +
                            (read_qword_keep[7:4] != 0 ? 2 : 1) == k_r) begin
                        w4_cache_valid[prefetch_cache_tile_index] <= 1'b1;
                        prefetch_active <= 1'b0;
                        prefetch_recv_index <= '0;
                    end else if (!prefetch_response_error) begin
                        prefetch_recv_index <= prefetch_recv_index +
                            (read_qword_keep[7:4] != 0 ? 2 : 1);
                    end
                end
                if (state == LOAD_A_PREP || state == LOAD_A_REQ || state == LOAD_A_RECV)
                    phase_load_a_cycles <= phase_load_a_cycles + 1'b1;
                if (state == LOAD_B_PREP || state == LOAD_B_REQ || state == LOAD_B_RECV ||
                    state == LOAD_BIAS_PREP || state == LOAD_BIAS_REQ || state == LOAD_BIAS_RECV ||
                    state == LOAD_BIAS_CACHE_PREFETCH || state == LOAD_BIAS_CACHE_RECV)
                    phase_load_b_cycles <= phase_load_b_cycles + 1'b1;
                if (state == LOAD_B_PREP || state == LOAD_B_REQ || state == LOAD_B_RECV)
                    phase_load_b_weight_cycles <= phase_load_b_weight_cycles + 1'b1;
                if (state == LOAD_BIAS_PREP || state == LOAD_BIAS_REQ || state == LOAD_BIAS_RECV ||
                    state == LOAD_BIAS_CACHE_PREFETCH || state == LOAD_BIAS_CACHE_RECV)
                    phase_load_b_bias_cycles <= phase_load_b_bias_cycles + 1'b1;
                if (state == CORE_START || state == CORE_PREFETCH ||
                    state == CORE_FEED || state == CORE_RESULT)
                    phase_compute_cycles <= phase_compute_cycles + 1'b1;
                if (fused_writer_busy || state == WRITE_RESULT || state == FUSED_START ||
                    state == FUSED_WAIT || state == FUSED_DRAIN ||
                    state == WRITE_ARM || state == WRITE_WAIT)
                    phase_write_cycles <= phase_write_cycles + 1'b1;
            end
            if (memory_error || core0_error || core1_error || fused_writer_error ||
                prefetch_response_error) begin error <= 1'b1; state <= FAIL; end
            else if (pipeline_enable) unique case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (dim_m == 0 || dim_n == 0 || dim_k == 0 || dim_k > MAX_K ||
                        // Exported DeiT W4 matrices occupy at most 18,432
                        // packed vectors: wide layers have K<=192 and deep
                        // layers have N<=192. Avoid a variable multiplier in
                        // this timing-sensitive launch/control cone.
                        (weight_int4 && !((dim_k <= 192 && dim_n <= MAX_N) ||
                                         (dim_k <= 768 && dim_n <= 192))) ||
                        a_base[2:0] != 0 || b_base[1:0] != 0 || c_base[1:0] != 0 ||
                        (fused_requant && (rq_right_shift > 31 ||
                            (fused_output_mode == 0 && dim_n[0]))) ||
                        (!fused_requant && (c_stride < dim_n * 4 || c_stride[1:0] != 0))) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        weight_int4_r <= weight_int4; fused_requant_r <= fused_requant;
                        fused_gelu_r <= fused_gelu;
                        fused_output_mode_r <= fused_output_mode;
                        a_base_r <= a_base; b_base_r <= b_base; bias_base_r <= bias_base;
                        c_base_r <= c_base; c_stride_r <= c_stride;
                        rq_multiplier_r <= rq_multiplier; rq_right_shift_r <= rq_right_shift;
                        rq_output_zero_point_r <= rq_output_zero_point;
                        m_r <= dim_m; n_r <= dim_n; k_r <= dim_k;
                        m_index <= 0; n_index <= 0; load_index <= 0; load_a_bank <= 1'b0;
                        w4_cache_valid <= '0; bias_cache_valid <= '0; w4_cache_tile_base <= '0;
                        prefetch_active <= 1'b0;
                        b_issue_index <= '0;
                        a_tile_addr <= a_base; b_tile_addr <= b_base; c_tile_addr <= c_base;
                        a_tile_stride <= dim_k << 3;
                        b_tile_stride <= weight_int4 ? (dim_k << 2) : (dim_k << 3);
                        c_m_pair_stride <= fused_requant ? (dim_n << 4) : (c_stride << 4);
                        valid_rows0 <= dim_m >= 8 ? 8 : dim_m[3:0];
                        valid_rows1 <= dim_m > 8
                            ? (dim_m - 8 >= 8 ? 8 : dim_m - 8) : 0;
                        second_core_active <= dim_m > 8;
                        valid_cols <= dim_n >= 8 ? 8 : dim_n[3:0];
                        state <= LOAD_A_PREP;
                    end
                end
                LOAD_A_PREP: begin
                    read_req_addr_r <= active_read_addr;
                    read_req_len_r <= prepared_read_len;
                    state <= LOAD_A_REQ;
                end
                LOAD_A_REQ: if (read_req_valid && read_req_ready) state <= LOAD_A_RECV;
                LOAD_A_RECV: if (read_qword_valid) begin
                    if (read_qword_keep != 8'hff ||
                        (load_index + 1'b1 == k_r && !read_qword_last)) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        if (load_index + 1'b1 == k_r) begin
                            load_index <= 0;
                            if (!load_a_bank && second_core_active) begin
                                load_a_bank <= 1'b1;
                                state <= LOAD_A_PREP;
                            end else begin
                                load_a_bank <= 1'b0;
                                state <= LOAD_B_PREP;
                            end
                        end
                        else begin
                            load_index <= load_index + 1'b1;
                            if (read_qword_last) state <= LOAD_A_PREP;
                        end
                    end
                end
                LOAD_B_PREP: begin
                    if (prefetch_active) begin
                        // The background stream owns the ordered response
                        // channel until this next tile is completely valid.
                    end else if (weight_int4_r && w4_cache_valid[w4_cache_tile_index]) begin
                        w4_cache_hit_tiles <= w4_cache_hit_tiles + 1'b1;
                        load_index <= 0;
                        state <= LOAD_BIAS_PREP;
                    end else begin
                        if (weight_int4_r)
                            w4_cache_miss_tiles <= w4_cache_miss_tiles + 1'b1;
                        b_issue_index <= 0;
                        read_req_addr_r <= active_read_addr;
                        read_req_len_r <= prepared_read_len;
                        state <= LOAD_B_REQ;
                    end
                end
                LOAD_B_REQ: if (read_req_valid && read_req_ready) begin
                    b_issue_index <= b_issue_index + (read_req_len >> (weight_int4_r ? 2 : 3));
                    state <= LOAD_B_RECV;
                end
                LOAD_B_RECV: begin
                    if (read_req_valid && read_req_ready)
                        b_issue_index <= b_issue_index + (read_req_len >> (weight_int4_r ? 2 : 3));
                    if (read_qword_valid) begin
                    if ((!weight_int4_r && read_qword_keep != 8'hff) ||
                        (weight_int4_r && read_qword_keep !=
                            (load_index + 1'b1 == k_r ? 8'h0f : 8'hff)) ||
                        (load_index + (weight_int4_r ?
                            (read_qword_keep[7:4] != 0 ? 2 : 1) : 1) == k_r &&
                            !read_qword_last)) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        if (load_index + (weight_int4_r ?
                            (read_qword_keep[7:4] != 0 ? 2 : 1) : 1) == k_r) begin
                            if (weight_int4_r)
                                w4_cache_valid[w4_cache_tile_index] <= 1'b1;
                            load_index <= 0;
                            state <= LOAD_BIAS_PREP;
                        end
                        else begin
                            load_index <= load_index + (weight_int4_r ? 2 : 1);
                        end
                    end
                    end
                end
                LOAD_BIAS_PREP: begin
                    if (!fused_requant_r || bias_base_r == 0) begin
                        if (weight_int4_r && m_index == 0 &&
                            n_index + valid_cols < n_r) begin
                            prefetch_active <= 1'b1;
                            prefetch_issue_index <= '0; prefetch_recv_index <= '0;
                            prefetch_b_addr <= b_tile_addr + b_tile_stride;
                            prefetch_cache_tile_base <= w4_cache_tile_base + ((k_r + 1'b1) >> 1);
                            prefetch_cache_tile_index <= w4_cache_tile_index + 1'b1;
                        end
                        load_index <= 0;
                        state <= CORE_START;
                    end else if (bias_cache_valid[w4_cache_tile_index]) begin
                        load_index <= 0;
                        state <= LOAD_BIAS_CACHE_PREFETCH;
                    end else begin
                        if (load_index == 0) bias_values <= '0;
                        read_req_addr_r <= active_read_addr;
                        read_req_len_r <= prepared_read_len;
                        state <= LOAD_BIAS_REQ;
                    end
                end
                LOAD_BIAS_REQ: if (read_req_valid && read_req_ready)
                    state <= LOAD_BIAS_RECV;
                LOAD_BIAS_RECV: if (read_qword_valid) begin
                    if (read_qword_keep != ((load_index << 1) + 1 < valid_cols ? 8'hff : 8'h0f) ||
                        !read_qword_last) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        bias_values[(load_index*2)*32 +: 32] <= read_qword_data[31:0];
                        if ((load_index << 1) + 1 < valid_cols)
                            bias_values[(load_index*2+1)*32 +: 32] <= read_qword_data[63:32];
                        if ((load_index << 1) + 2 >= valid_cols) begin
                            bias_cache_valid[w4_cache_tile_index] <= 1'b1;
                            if (weight_int4_r && m_index == 0 &&
                                n_index + valid_cols < n_r) begin
                                prefetch_active <= 1'b1;
                                prefetch_issue_index <= '0; prefetch_recv_index <= '0;
                                prefetch_b_addr <= b_tile_addr + b_tile_stride;
                                prefetch_cache_tile_base <= w4_cache_tile_base + ((k_r + 1'b1) >> 1);
                                prefetch_cache_tile_index <= w4_cache_tile_index + 1'b1;
                            end
                            load_index <= 0; state <= CORE_START;
                        end else begin
                            load_index <= load_index + 1'b1;
                            state <= LOAD_BIAS_PREP;
                        end
                    end
                end
                LOAD_BIAS_CACHE_PREFETCH: state <= LOAD_BIAS_CACHE_RECV;
                LOAD_BIAS_CACHE_RECV: begin
                    bias_values[(load_index*2)*32 +: 64] <= bias_cache_qword;
                    if ((load_index << 1) + 2 >= valid_cols) begin
                        if (weight_int4_r && m_index == 0 &&
                            n_index + valid_cols < n_r) begin
                            prefetch_active <= 1'b1;
                            prefetch_issue_index <= '0; prefetch_recv_index <= '0;
                            prefetch_b_addr <= b_tile_addr + b_tile_stride;
                            prefetch_cache_tile_base <= w4_cache_tile_base + ((k_r + 1'b1) >> 1);
                            prefetch_cache_tile_index <= w4_cache_tile_index + 1'b1;
                        end
                        load_index <= 0;
                        state <= CORE_START;
                    end else begin
                        load_index <= load_index + 1'b1;
                        state <= LOAD_BIAS_CACHE_PREFETCH;
                    end
                end
                CORE_START: begin
                    feed_index <= 0;
                    state <= CORE_PREFETCH;
                end
                CORE_PREFETCH: begin
                    state <= CORE_FEED;
                end
                CORE_FEED: if (core0_input_valid && core0_input_ready &&
                    (!second_core_active || (core1_input_valid && core1_input_ready))) begin
                    if (feed_index + 1'b1 == k_r) state <= CORE_RESULT;
                    else begin
                        feed_index <= feed_index + 1'b1;
                    end
                end
                CORE_RESULT: if (core0_result_valid &&
                    (!second_core_active || core1_result_valid)) begin
                    write_core_select <= 1'b0;
                    write_row <= 0; write_col <= 0; write_addr_r <= c_tile_addr;
                    state <= fused_requant_r ? FUSED_START : WRITE_RESULT;
                end
                WRITE_RESULT: if (write_req_valid && write_req_ready) begin
                    if (write_col + 1'b1 == valid_cols) begin
                        write_col <= 0;
                        if (write_row + 1'b1 ==
                            (write_core_select ? valid_rows1 : valid_rows0)) begin
                            if (!write_core_select && second_core_active) begin
                                write_core_select <= 1'b1;
                                write_row <= 0;
                                write_addr_r <= c_tile_addr + (c_stride_r << 3);
                            end else state <= WRITE_ARM;
                        end
                        else begin write_row <= write_row + 1'b1; write_addr_r <= write_addr_r + c_stride_r - ((valid_cols - 1'b1) << 2); end
                    end else begin write_col <= write_col + 1'b1; write_addr_r <= write_addr_r + 4; end
                end
                FUSED_START: if (!fused_writer_busy) state <= FUSED_WAIT;
                FUSED_WAIT: if (fused_writer_capture_done) begin
                    if (!write_core_select && second_core_active) begin
                        write_core_select <= 1'b1;
                        state <= FUSED_DRAIN;
                    end else state <= NEXT_TILE;
                end
                FUSED_DRAIN: if (fused_writer_done) state <= FUSED_START;
                WRITE_ARM: state <= WRITE_WAIT;
                WRITE_WAIT: if (!fused_writer_busy && !memory_busy)
                    state <= fused_requant_r ? FINISH : NEXT_TILE;
                NEXT_TILE: begin
                    if (n_index + valid_cols >= n_r) begin
                        n_index <= 0;
                        if (m_index + valid_rows0 + valid_rows1 >= m_r)
                            state <= fused_requant_r ? WRITE_ARM : FINISH;
                        else begin
                            m_index <= m_index + 16;
                            a_tile_addr <= a_tile_addr + (a_tile_stride << 1);
                            b_tile_addr <= b_base_r;
                            if (fused_requant_r && fused_output_mode_r == 2'd2)
                                c_tile_addr <= c_base_r + ((m_index + 16) << 3);
                            else c_tile_addr <= c_tile_addr + c_m_pair_stride -
                                (fused_requant_r ?
                                    (fused_output_mode_r == 2'd1 ? n_index : (n_index << 3)) :
                                    (n_index << 2));
                            valid_rows0 <= m_r - (m_index + 16) >= 8
                                ? 8 : m_r - (m_index + 16);
                            valid_rows1 <= m_r - (m_index + 16) > 8
                                ? (m_r - (m_index + 16) >= 16
                                    ? 8 : m_r - (m_index + 24)) : 0;
                            second_core_active <= m_r - (m_index + 16) > 8;
                            valid_cols <= n_r >= 8 ? 8 : n_r[3:0];
                            w4_cache_tile_base <= '0;
                            load_index <= 0; load_a_bank <= 1'b0; state <= LOAD_A_PREP;
                        end
                    end else begin
                        n_index <= n_index + 8;
                        w4_cache_tile_base <= w4_cache_tile_base + ((k_r + 1'b1) >> 1);
                        b_tile_addr <= b_tile_addr + b_tile_stride;
                        c_tile_addr <= c_tile_addr + (fused_requant_r ?
                            (fused_output_mode_r == 2'd1 ? 8 :
                             fused_output_mode_r == 2'd2 ? (m_r << 3) : 64) : 32);
                        valid_cols <= n_r - (n_index + 8) >= 8 ? 8 : n_r - (n_index + 8);
                        /* activation_ram still contains this M tile; retain it
                         * while walking every N tile. */
                        load_index <= 0; state <= LOAD_B_PREP;
                    end
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; if (!start) state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
