module transformer_gemm_scheduler #(
    parameter int MAX_K = 1024,
    parameter int K_WIDTH = $clog2(MAX_K + 1)
) (
    input  logic clk, rst_n, pipeline_enable,
    input  logic start, weight_int4,
    input  logic [31:0] a_base, b_base, c_base, c_stride,
    input  logic [15:0] dim_m, dim_n, dim_k,
    output logic busy, done, error,
    output logic op_start, op_done,
    output logic memory_stall, input_stall, output_stall,
    output logic [31:0] debug_state,

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
        CORE_START, CORE_PREFETCH, CORE_FEED, CORE_RESULT, WRITE_RESULT,
        WRITE_ARM, WRITE_WAIT, NEXT_TILE, FINISH, FAIL
    } state_t;
    state_t state;

    (* ram_style = "block" *) logic [63:0] activation_ram0 [0:MAX_K-1];
    (* ram_style = "block" *) logic [63:0] activation_ram1 [0:MAX_K-1];
    (* ram_style = "block" *) logic [63:0] weight_ram [0:MAX_K-1];
    logic [63:0] feed_activation0, feed_activation1, feed_weight;
    logic [K_WIDTH-1:0] load_index, feed_index;
    logic [15:0] m_index, n_index;
    logic [3:0] valid_rows0, valid_rows1, valid_cols, write_row, write_col;
    logic load_a_bank, write_core_select, second_core_active;
    logic weight_int4_r;
    logic [31:0] a_base_r, b_base_r, c_base_r, c_stride_r;
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
    logic [K_WIDTH-1:0] ram_read_index;

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
    assign core0_result_ready = state == NEXT_TILE && pipeline_enable;
    assign core1_result_ready = core0_result_ready && second_core_active;
    assign feed_advance = core0_input_valid && core0_input_ready &&
        (!second_core_active || (core1_input_valid && core1_input_ready));

    assign read_req_valid = pipeline_enable && (state == LOAD_A_REQ || state == LOAD_B_REQ);
    assign active_read_addr = state == LOAD_A_PREP
        ? a_tile_addr + (load_a_bank ? a_tile_stride : 0) + (load_index << 3)
        : b_tile_addr + (weight_int4_r ? (load_index << 2) : (load_index << 3));
    assign active_read_bytes_remaining = (k_r - load_index) *
        ((state == LOAD_B_PREP && weight_int4_r) ? 17'd4 : 17'd8);
    assign active_read_line_remaining = 7'd64 - {1'b0, active_read_addr[5:0]};
    assign read_req_addr = read_req_addr_r;
    /* W8 vectors are streamed a cache line at a time. Packed W4 retains its
     * one-vector request until the wide W4 splitter is connected here. */
    assign read_req_len = read_req_len_r;
    wire [6:0] prepared_read_len = state == LOAD_B_PREP && weight_int4_r ? 7'd4 :
        active_read_bytes_remaining < active_read_line_remaining
            ? active_read_bytes_remaining[6:0] : active_read_line_remaining;
    assign read_req_wide = 1'b1;
    assign read_req_qword = 1'b1;
    assign read_req_allow_ahead = 1'b0;
    assign read_qword_ready = pipeline_enable && (state == LOAD_A_RECV || state == LOAD_B_RECV);

    assign write_req_valid = state == WRITE_RESULT && pipeline_enable;
    assign write_req_addr = write_addr_r;
    assign write_req_data = write_core_select
        ? accumulators1[(write_row*8+write_col)*32 +: 32]
        : accumulators0[(write_row*8+write_col)*32 +: 32];
    assign write_req_strb = 4'hf;
    assign write_req_last = write_col + 1'b1 == valid_cols;

    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
                                   (read_qword_ready && !read_qword_valid) ||
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
    assign weight_write_en = state == LOAD_B_RECV && read_qword_valid && read_qword_ready;
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
        if (weight_write_en)
            weight_ram[load_index] <= weight_int4_r
                ? {32'd0, read_qword_data[31:0]} : read_qword_data;
        feed_weight <= weight_ram[ram_read_index];
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

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            load_index <= '0; feed_index <= '0; m_index <= '0; n_index <= '0;
            valid_rows0 <= '0; valid_rows1 <= '0; valid_cols <= '0;
            write_row <= '0; write_col <= '0; load_a_bank <= 1'b0;
            write_core_select <= 1'b0; second_core_active <= 1'b0;
            weight_int4_r <= 1'b0; a_base_r <= '0; b_base_r <= '0;
            c_base_r <= '0; c_stride_r <= '0; m_r <= '0; n_r <= '0; k_r <= '0;
            a_tile_addr <= '0; b_tile_addr <= '0; c_tile_addr <= '0;
            a_tile_stride <= '0; b_tile_stride <= '0; c_m_pair_stride <= '0; write_addr_r <= '0;
            read_req_addr_r <= '0; read_req_len_r <= '0;
        end else begin
            done <= 1'b0;
            if (memory_error || core0_error || core1_error) begin error <= 1'b1; state <= FAIL; end
            else if (pipeline_enable) unique case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (dim_m == 0 || dim_n == 0 || dim_k == 0 || dim_k > MAX_K ||
                        a_base[2:0] != 0 || b_base[1:0] != 0 || c_base[1:0] != 0 ||
                        c_stride < dim_n * 4 || c_stride[1:0] != 0) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        weight_int4_r <= weight_int4; a_base_r <= a_base; b_base_r <= b_base;
                        c_base_r <= c_base; c_stride_r <= c_stride;
                        m_r <= dim_m; n_r <= dim_n; k_r <= dim_k;
                        m_index <= 0; n_index <= 0; load_index <= 0; load_a_bank <= 1'b0;
                        a_tile_addr <= a_base; b_tile_addr <= b_base; c_tile_addr <= c_base;
                        a_tile_stride <= dim_k << 3;
                        b_tile_stride <= weight_int4 ? (dim_k << 2) : (dim_k << 3);
                        c_m_pair_stride <= c_stride << 4;
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
                    read_req_addr_r <= active_read_addr;
                    read_req_len_r <= prepared_read_len;
                    state <= LOAD_B_REQ;
                end
                LOAD_B_REQ: if (read_req_valid && read_req_ready) state <= LOAD_B_RECV;
                LOAD_B_RECV: if (read_qword_valid) begin
                    if (read_qword_keep != (weight_int4_r ? 8'h0f : 8'hff) ||
                        (weight_int4_r && !read_qword_last) ||
                        (load_index + 1'b1 == k_r && !read_qword_last)) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        if (load_index + 1'b1 == k_r) begin load_index <= 0; state <= CORE_START; end
                        else begin
                            load_index <= load_index + 1'b1;
                            if (read_qword_last) state <= LOAD_B_PREP;
                        end
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
                    write_row <= 0; write_col <= 0; write_addr_r <= c_tile_addr; state <= WRITE_RESULT;
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
                WRITE_ARM: state <= WRITE_WAIT;
                WRITE_WAIT: if (!memory_busy) state <= NEXT_TILE;
                NEXT_TILE: begin
                    if (n_index + valid_cols >= n_r) begin
                        n_index <= 0;
                        if (m_index + valid_rows0 + valid_rows1 >= m_r) state <= FINISH;
                        else begin
                            m_index <= m_index + 16;
                            a_tile_addr <= a_tile_addr + (a_tile_stride << 1);
                            b_tile_addr <= b_base_r;
                            c_tile_addr <= c_tile_addr + c_m_pair_stride - (n_index << 2);
                            valid_rows0 <= m_r - (m_index + 16) >= 8
                                ? 8 : m_r - (m_index + 16);
                            valid_rows1 <= m_r - (m_index + 16) > 8
                                ? (m_r - (m_index + 16) >= 16
                                    ? 8 : m_r - (m_index + 24)) : 0;
                            second_core_active <= m_r - (m_index + 16) > 8;
                            valid_cols <= n_r >= 8 ? 8 : n_r[3:0];
                            load_index <= 0; load_a_bank <= 1'b0; state <= LOAD_A_PREP;
                        end
                    end else begin
                        n_index <= n_index + 8;
                        b_tile_addr <= b_tile_addr + b_tile_stride;
                        c_tile_addr <= c_tile_addr + 32;
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
