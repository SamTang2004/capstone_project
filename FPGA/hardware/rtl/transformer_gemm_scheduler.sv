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
    typedef enum logic [3:0] {
        IDLE, LOAD_A_REQ, LOAD_A_RECV, LOAD_B_REQ, LOAD_B_RECV,
        CORE_START, CORE_PREFETCH, CORE_FEED, CORE_RESULT, WRITE_RESULT,
        WRITE_ARM, WRITE_WAIT, NEXT_TILE, FINISH, FAIL
    } state_t;
    state_t state;

    (* ram_style = "block" *) logic [63:0] activation_ram [0:MAX_K-1];
    (* ram_style = "block" *) logic [63:0] weight_ram [0:MAX_K-1];
    logic [63:0] feed_activation, feed_weight;
    logic [K_WIDTH-1:0] load_index, feed_index;
    logic [15:0] m_index, n_index;
    logic [3:0] valid_rows, valid_cols, write_row, write_col;
    logic weight_int4_r;
    logic [31:0] a_base_r, b_base_r, c_base_r, c_stride_r;
    logic [31:0] a_tile_addr, b_tile_addr, c_tile_addr;
    logic [31:0] a_tile_stride, b_tile_stride, c_m_tile_stride, write_addr_r;
    logic [15:0] m_r, n_r, k_r;
    logic core_start, core_busy, core_input_valid, core_input_ready;
    logic core_result_valid, core_result_ready, core_error;
    logic [2047:0] accumulators;
    logic [31:0] weight_bytes;

    assign weight_bytes = weight_int4_r ? 32'd4 : 32'd8;
    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;
    assign core_start = state == CORE_START && pipeline_enable;
    assign core_input_valid = state == CORE_FEED && pipeline_enable;
    assign core_result_ready = state == NEXT_TILE && pipeline_enable;

    assign read_req_valid = pipeline_enable && (state == LOAD_A_REQ || state == LOAD_B_REQ);
    assign read_req_addr = state == LOAD_A_REQ
        ? a_tile_addr + (load_index << 3)
        : b_tile_addr + (weight_int4_r ? (load_index << 2) : (load_index << 3));
    assign read_req_len = state == LOAD_A_REQ || !weight_int4_r ? 7'd8 : 7'd4;
    assign read_req_wide = 1'b1;
    assign read_req_qword = 1'b1;
    assign read_req_allow_ahead = 1'b0;
    assign read_qword_ready = pipeline_enable && (state == LOAD_A_RECV || state == LOAD_B_RECV);

    assign write_req_valid = state == WRITE_RESULT && pipeline_enable;
    assign write_req_addr = write_addr_r;
    assign write_req_data = accumulators[(write_row*8+write_col)*32 +: 32];
    assign write_req_strb = 4'hf;
    assign write_req_last = write_col + 1'b1 == valid_cols;

    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
                                   (read_qword_ready && !read_qword_valid) ||
                                   (write_req_valid && !write_req_ready));
    assign input_stall = state == CORE_FEED && !core_input_ready;
    assign output_stall = state == CORE_RESULT && !core_result_valid;
    assign debug_state = {16'd0, core_error, core_result_valid, core_input_ready,
                          memory_busy, 4'd0, state};

    gemm_8x8_tile_core core (
        .clk, .rst_n, .start(core_start), .weight_int4(weight_int4_r),
        .k_steps(k_r), .valid_rows, .valid_cols,
        .input_valid(core_input_valid), .input_ready(core_input_ready),
        .activation_vector(feed_activation), .packed_weight_vector(feed_weight),
        .busy(core_busy), .result_valid(core_result_valid),
        .result_ready(core_result_ready), .error(core_error), .accumulators
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            load_index <= '0; feed_index <= '0; m_index <= '0; n_index <= '0;
            valid_rows <= '0; valid_cols <= '0; write_row <= '0; write_col <= '0;
            weight_int4_r <= 1'b0; a_base_r <= '0; b_base_r <= '0;
            c_base_r <= '0; c_stride_r <= '0; m_r <= '0; n_r <= '0; k_r <= '0;
            a_tile_addr <= '0; b_tile_addr <= '0; c_tile_addr <= '0;
            a_tile_stride <= '0; b_tile_stride <= '0; c_m_tile_stride <= '0; write_addr_r <= '0;
        end else begin
            done <= 1'b0;
            if (memory_error || core_error) begin error <= 1'b1; state <= FAIL; end
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
                        m_index <= 0; n_index <= 0; load_index <= 0;
                        a_tile_addr <= a_base; b_tile_addr <= b_base; c_tile_addr <= c_base;
                        a_tile_stride <= dim_k << 3;
                        b_tile_stride <= weight_int4 ? (dim_k << 2) : (dim_k << 3);
                        c_m_tile_stride <= c_stride << 3;
                        valid_rows <= dim_m >= 8 ? 8 : dim_m[3:0];
                        valid_cols <= dim_n >= 8 ? 8 : dim_n[3:0];
                        state <= LOAD_A_REQ;
                    end
                end
                LOAD_A_REQ: if (read_req_valid && read_req_ready) state <= LOAD_A_RECV;
                LOAD_A_RECV: if (read_qword_valid) begin
                    if (read_qword_keep != 8'hff || !read_qword_last) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        activation_ram[load_index] <= read_qword_data;
                        if (load_index + 1'b1 == k_r) begin load_index <= 0; state <= LOAD_B_REQ; end
                        else begin load_index <= load_index + 1'b1; state <= LOAD_A_REQ; end
                    end
                end
                LOAD_B_REQ: if (read_req_valid && read_req_ready) state <= LOAD_B_RECV;
                LOAD_B_RECV: if (read_qword_valid) begin
                    if (read_qword_keep != (weight_int4_r ? 8'h0f : 8'hff) || !read_qword_last) begin error <= 1'b1; state <= FAIL; end
                    else begin
                        weight_ram[load_index] <= weight_int4_r ? {32'd0, read_qword_data[31:0]} : read_qword_data;
                        if (load_index + 1'b1 == k_r) begin load_index <= 0; state <= CORE_START; end
                        else begin load_index <= load_index + 1'b1; state <= LOAD_B_REQ; end
                    end
                end
                CORE_START: begin
                    feed_index <= 0;
                    state <= CORE_PREFETCH;
                end
                CORE_PREFETCH: begin
                    feed_activation <= activation_ram[feed_index];
                    feed_weight <= weight_ram[feed_index];
                    state <= CORE_FEED;
                end
                CORE_FEED: if (core_input_valid && core_input_ready) begin
                    if (feed_index + 1'b1 == k_r) state <= CORE_RESULT;
                    else begin
                        feed_index <= feed_index + 1'b1;
                        state <= CORE_PREFETCH;
                    end
                end
                CORE_RESULT: if (core_result_valid) begin
                    write_row <= 0; write_col <= 0; write_addr_r <= c_tile_addr; state <= WRITE_RESULT;
                end
                WRITE_RESULT: if (write_req_valid && write_req_ready) begin
                    if (write_col + 1'b1 == valid_cols) begin
                        write_col <= 0;
                        if (write_row + 1'b1 == valid_rows) state <= WRITE_ARM;
                        else begin write_row <= write_row + 1'b1; write_addr_r <= write_addr_r + c_stride_r - ((valid_cols - 1'b1) << 2); end
                    end else begin write_col <= write_col + 1'b1; write_addr_r <= write_addr_r + 4; end
                end
                WRITE_ARM: state <= WRITE_WAIT;
                WRITE_WAIT: if (!memory_busy) state <= NEXT_TILE;
                NEXT_TILE: begin
                    if (n_index + valid_cols >= n_r) begin
                        n_index <= 0;
                        if (m_index + valid_rows >= m_r) state <= FINISH;
                        else begin
                            m_index <= m_index + 8;
                            a_tile_addr <= a_tile_addr + a_tile_stride;
                            b_tile_addr <= b_base_r;
                            c_tile_addr <= c_tile_addr + c_m_tile_stride - (n_index << 2);
                            valid_rows <= m_r - (m_index + 8) >= 8 ? 8 : m_r - (m_index + 8);
                            valid_cols <= n_r >= 8 ? 8 : n_r[3:0];
                            load_index <= 0; state <= LOAD_A_REQ;
                        end
                    end else begin
                        n_index <= n_index + 8;
                        b_tile_addr <= b_tile_addr + b_tile_stride;
                        c_tile_addr <= c_tile_addr + 32;
                        valid_cols <= n_r - (n_index + 8) >= 8 ? 8 : n_r - (n_index + 8);
                        load_index <= 0; state <= LOAD_A_REQ;
                    end
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; if (!start) state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
