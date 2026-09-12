module llm_gemm_ddr_top (
    input logic aclk, aresetn,
    input logic [7:0] s_axi_awaddr, input logic s_axi_awvalid, output logic s_axi_awready,
    input logic [31:0] s_axi_wdata, input logic [3:0] s_axi_wstrb, input logic s_axi_wvalid, output logic s_axi_wready,
    output logic [1:0] s_axi_bresp, output logic s_axi_bvalid, input logic s_axi_bready,
    input logic [7:0] s_axi_araddr, input logic s_axi_arvalid, output logic s_axi_arready,
    output logic [31:0] s_axi_rdata, output logic [1:0] s_axi_rresp, output logic s_axi_rvalid, input logic s_axi_rready,
    output logic mem_cmd_valid, input logic mem_cmd_ready, output logic mem_cmd_write,
    output logic [31:0] mem_cmd_addr, output logic [3:0] mem_cmd_len,
    output logic [31:0] mem_cmd_wdata, output logic [3:0] mem_cmd_wstrb,
    output logic mem_write_valid, input logic mem_write_ready,
    output logic [31:0] mem_write_data, output logic [3:0] mem_write_strb, output logic mem_write_last,
    input logic mem_read_valid, output logic mem_read_ready, input logic [63:0] mem_read_data, input logic mem_read_last,
    input logic [3:0] mem_debug_arlen, input logic [2:0] mem_debug_arsize,
    input logic [4:0] mem_debug_rbeats, input logic [1:0] mem_debug_rresp
);
    logic [31:0] ctrl_word, debug_ctrl_word, status_word;
    logic [32*16-1:0] regs_flat, perf_regs_flat;
    logic [32*18-1:0] debug_regs_flat;
    logic [32*6-1:0] extended_perf_regs_flat;
    logic start_d, start_pulse, unsupported_cmd;
    logic scheduler_busy, scheduler_done, scheduler_error;
    logic accelerator_busy, accelerator_done, accelerator_error;
    logic done_latched, error_latched, pipeline_enable;
    logic op_start, op_done, scheduler_memory_stall, input_stall, output_stall;

    logic g_read_req_valid, g_read_req_ready, g_read_req_wide, g_read_req_qword, g_read_req_allow_ahead;
    logic [31:0] g_read_req_addr; logic [6:0] g_read_req_len;
    logic g_read_qword_ready, g_write_req_valid, g_write_req_ready, g_write_req_last;
    logic [31:0] g_write_req_addr, g_write_req_data; logic [3:0] g_write_req_strb;
    logic g_op_start, g_op_done, g_memory_stall, g_input_stall, g_output_stall;
    logic read_req_valid, read_req_ready, read_req_wide, read_req_qword, read_req_allow_ahead;
    logic [31:0] read_req_addr; logic [6:0] read_req_len;
    logic arb_read_req_valid, arb_read_req_wide, arb_read_req_qword, arb_read_req_allow_ahead;
    logic [31:0] arb_read_req_addr; logic [6:0] arb_read_req_len;
    logic read_req_pipe_ready;
    logic read_data_valid, read_data_ready, read_data_last; logic [7:0] read_data;
    logic read_word_valid, read_word_ready, read_word_last; logic [31:0] read_word_data; logic [3:0] read_word_keep;
    logic read_qword_valid, read_qword_ready, read_qword_last; logic [63:0] read_qword_data; logic [7:0] read_qword_keep;
    logic write_req_valid, write_req_ready, write_req_last; logic [31:0] write_req_addr, write_req_data; logic [3:0] write_req_strb;
    logic arb_write_req_valid, arb_write_req_last;
    logic [31:0] arb_write_req_addr, arb_write_req_data;
    logic [3:0] arb_write_req_strb;
    logic write_req_pipe_ready;
    logic memory_busy, memory_error; logic [2:0] memory_error_code; logic [7:0] debug_first_byte; logic [4:0] debug_fill_beats;
    logic [31:0] stats_cache_hits, stats_cache_misses, stats_fill_cycles, stats_bytes_delivered;
    logic [31:0] scheduler_debug_state, memory_debug_state;
    logic [63:0] gemm_load_a_cycles, gemm_load_b_cycles;
    logic [63:0] gemm_load_b_weight_cycles, gemm_load_b_bias_cycles;
    logic [63:0] gemm_w4_cache_hit_tiles, gemm_w4_cache_miss_tiles;
    logic [63:0] gemm_compute_cycles, gemm_write_cycles;

    wire [31:0] a_base = regs_flat[2*32 +: 32];
    wire [31:0] b_base = regs_flat[3*32 +: 32];
    wire [31:0] c_base = regs_flat[4*32 +: 32];
    wire [31:0] c_stride = regs_flat[7*32 +: 32];
    wire [15:0] dim_m = regs_flat[8*32 +: 16];
    wire [15:0] dim_n = regs_flat[9*32 +: 16];
    wire [15:0] dim_k = regs_flat[10*32 +: 16];
    assign start_pulse = ctrl_word[0] && !start_d;
    assign unsupported_cmd = start_pulse && ((ctrl_word & ~32'h00000801) != 0);
    assign accelerator_busy = scheduler_busy;
    assign accelerator_done = scheduler_done;
    assign accelerator_error = scheduler_error || unsupported_cmd;
    assign status_word = {24'd0, memory_error_code, memory_error, 1'b0, error_latched, done_latched, accelerator_busy};
    assign op_start = g_op_start;
    assign op_done = g_op_done;
    assign scheduler_memory_stall = g_memory_stall;
    assign input_stall = g_input_stall;
    assign output_stall = g_output_stall;
    always_comb begin
        perf_regs_flat = '0;
        perf_regs_flat[0*32 +: 32] = scheduler_debug_state;
        perf_regs_flat[1*32 +: 32] = memory_debug_state;
        perf_regs_flat[2*32 +: 64] = gemm_load_a_cycles;
        perf_regs_flat[4*32 +: 64] = gemm_load_b_cycles;
        perf_regs_flat[6*32 +: 64] = gemm_compute_cycles;
        perf_regs_flat[8*32 +: 64] = gemm_write_cycles;
        perf_regs_flat[10*32 +: 32] = {23'd0, mem_debug_arsize, mem_debug_arlen};
        perf_regs_flat[11*32 +: 32] = {25'd0, mem_debug_rresp, mem_debug_rbeats};
        perf_regs_flat[12*32 +: 32] = stats_cache_hits;
        perf_regs_flat[13*32 +: 32] = stats_cache_misses;
        perf_regs_flat[14*32 +: 32] = stats_fill_cycles;
        perf_regs_flat[15*32 +: 32] = stats_bytes_delivered;
        extended_perf_regs_flat = '0;
        extended_perf_regs_flat[0*32 +: 64] = gemm_load_b_weight_cycles;
        extended_perf_regs_flat[2*32 +: 64] = gemm_load_b_bias_cycles;
        extended_perf_regs_flat[4*32 +: 32] = gemm_w4_cache_hit_tiles[31:0];
        extended_perf_regs_flat[5*32 +: 32] = gemm_w4_cache_miss_tiles[31:0];
    end

    always_ff @(posedge aclk) begin
        if(!aresetn) begin start_d<=0;done_latched<=0;error_latched<=0; end
        else begin
            start_d <= ctrl_word[0];
            if(start_pulse) begin done_latched<=0;error_latched<=0; end
            if(accelerator_done) done_latched<=1;
            if(accelerator_error||memory_error) error_latched<=1;
            if(!ctrl_word[0]&&!accelerator_busy) done_latched<=0;
        end
    end

    axi_lite_control control (
        .aclk, .aresetn, .s_axi_awaddr, .s_axi_awvalid, .s_axi_awready,
        .s_axi_wdata, .s_axi_wstrb, .s_axi_wvalid, .s_axi_wready,
        .s_axi_bresp, .s_axi_bvalid, .s_axi_bready,
        .s_axi_araddr, .s_axi_arvalid, .s_axi_arready,
        .s_axi_rdata, .s_axi_rresp, .s_axi_rvalid, .s_axi_rready,
        .status_word, .perf_regs_flat, .debug_regs_flat, .extended_perf_regs_flat,
        .ctrl_word, .debug_ctrl_word, .regs_flat
    );

    transformer_control_status debug_status (
        .clk(aclk), .rst_n(aresetn), .debug_ctrl_word,
        .accelerator_busy, .memory_stall(scheduler_memory_stall),
        .input_stall, .output_stall, .op_start, .op_done,
        .op_code(8'd1),
        .error_event(accelerator_error||memory_error), .debug_data({28'd0,memory_error,memory_error_code}),
        .pipeline_enable, .debug_regs_flat
    );

    transformer_gemm_scheduler #(.ENABLE_FUSED_WRITER(1'b0)) scheduler (
        .clk(aclk), .rst_n(aresetn), .pipeline_enable, .start(start_pulse && !unsupported_cmd),
        .weight_int4(ctrl_word[11]), .fused_requant(1'b0),
        .fused_gelu(1'b0), .fused_output_mode(2'd0),
        .a_base, .b_base, .c_base, .c_stride, .bias_base(32'd0),
        .rq_multiplier(32'd0), .rq_right_shift(6'd0), .rq_output_zero_point(32'd0),
        .dim_m, .dim_n, .dim_k, .busy(scheduler_busy), .done(scheduler_done), .error(scheduler_error),
        .op_start(g_op_start), .op_done(g_op_done), .memory_stall(g_memory_stall), .input_stall(g_input_stall), .output_stall(g_output_stall),
        .debug_state(scheduler_debug_state),
        .phase_load_a_cycles(gemm_load_a_cycles), .phase_load_b_cycles(gemm_load_b_cycles),
        .phase_load_b_weight_cycles(gemm_load_b_weight_cycles),
        .phase_load_b_bias_cycles(gemm_load_b_bias_cycles),
        .w4_cache_hit_tiles(gemm_w4_cache_hit_tiles),
        .w4_cache_miss_tiles(gemm_w4_cache_miss_tiles),
        .phase_compute_cycles(gemm_compute_cycles), .phase_write_cycles(gemm_write_cycles),
        .read_req_valid(g_read_req_valid), .read_req_ready(g_read_req_ready), .read_req_addr(g_read_req_addr), .read_req_len(g_read_req_len),
        .read_req_wide(g_read_req_wide), .read_req_qword(g_read_req_qword), .read_req_allow_ahead(g_read_req_allow_ahead),
        .read_qword_valid, .read_qword_ready(g_read_qword_ready), .read_qword_data, .read_qword_keep, .read_qword_last,
        .write_req_valid(g_write_req_valid), .write_req_ready(g_write_req_ready), .write_req_addr(g_write_req_addr),
        .write_req_data(g_write_req_data), .write_req_strb(g_write_req_strb), .write_req_last(g_write_req_last),
        .memory_busy, .memory_error(memory_error)
    );

    assign read_req_pipe_ready = !read_req_valid || read_req_ready;
    assign write_req_pipe_ready = !write_req_valid || write_req_ready;
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            read_req_valid <= 1'b0;
            read_req_addr <= '0; read_req_len <= '0;
            read_req_wide <= 1'b0; read_req_qword <= 1'b0;
            read_req_allow_ahead <= 1'b0;
        end else if (read_req_pipe_ready) begin
            read_req_valid <= arb_read_req_valid;
            if (arb_read_req_valid) begin
                read_req_addr <= arb_read_req_addr; read_req_len <= arb_read_req_len;
                read_req_wide <= arb_read_req_wide; read_req_qword <= arb_read_req_qword;
                read_req_allow_ahead <= arb_read_req_allow_ahead;
            end
        end
    end
    // Register the selected producer's complete write request before the
    // shared DDR driver.  Without this one-entry skid stage, producer address
    // generation, arbitration, contiguous-burst comparison, cache
    // invalidation, and write-cache enables formed one 13-level path.  The
    // stage accepts a replacement in the same cycle that the previous entry
    // is consumed, so sustained write throughput remains one word per cycle.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            write_req_valid <= 1'b0;
            write_req_addr <= '0;
            write_req_data <= '0;
            write_req_strb <= '0;
            write_req_last <= 1'b0;
        end else if (write_req_pipe_ready) begin
            write_req_valid <= arb_write_req_valid;
            if (arb_write_req_valid) begin
                write_req_addr <= arb_write_req_addr;
                write_req_data <= arb_write_req_data;
                write_req_strb <= arb_write_req_strb;
                write_req_last <= arb_write_req_last;
            end
        end
    end
    assign arb_read_req_valid = g_read_req_valid;
    assign arb_read_req_addr = g_read_req_addr;
    assign arb_read_req_len = g_read_req_len;
    assign arb_read_req_wide = g_read_req_wide;
    assign arb_read_req_qword = g_read_req_qword;
    assign arb_read_req_allow_ahead = g_read_req_allow_ahead;
    assign arb_write_req_valid = g_write_req_valid;
    assign arb_write_req_addr = g_write_req_addr;
    assign arb_write_req_data = g_write_req_data;
    assign arb_write_req_strb = g_write_req_strb;
    assign arb_write_req_last = g_write_req_last;
    assign g_read_req_ready = read_req_pipe_ready;
    assign g_write_req_ready = write_req_pipe_ready;
    assign read_qword_ready = g_read_qword_ready;
    assign read_word_ready = 1'b0;
    assign read_data_ready = 1'b1;
    memory_io_driver #(.READ_AHEAD_ENABLE(1'b0),.WIDE_RESPONSE_ENABLE(1'b1)) memory (
        .clk(aclk), .rst_n(aresetn),
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_req_wide, .read_req_qword, .read_req_allow_ahead,
        .read_data_valid, .read_data_ready, .read_data, .read_data_last,
        .read_word_valid, .read_word_ready, .read_word_data, .read_word_keep, .read_word_last,
        .read_qword_valid, .read_qword_ready, .read_qword_data, .read_qword_keep, .read_qword_last,
        .write_req_valid, .write_req_ready, .write_req_addr, .write_req_data, .write_req_strb, .write_req_last,
        .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write, .mem_cmd_addr, .mem_cmd_len, .mem_cmd_wdata, .mem_cmd_wstrb,
        .mem_write_valid, .mem_write_ready, .mem_write_data, .mem_write_strb, .mem_write_last,
        .mem_read_valid, .mem_read_ready, .mem_read_data, .mem_read_last,
        .busy(memory_busy), .error(memory_error), .error_code(memory_error_code),
        .debug_first_byte, .debug_fill_beats, .cache_invalidate(start_pulse), .stats_reset(start_pulse),
        .stats_cache_hits, .stats_cache_misses, .stats_fill_cycles, .stats_bytes_delivered,
        .debug_state(memory_debug_state)
    );
endmodule
