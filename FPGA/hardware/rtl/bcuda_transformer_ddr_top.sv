module bcuda_transformer_ddr_top #(
    parameter bit ENABLE_LEGACY_STANDALONE = 1'b1
) (
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
    logic start_d, start_pulse, active_requant, active_pack, active_b_pack, active_elementwise, active_rowwise, active_prune;
    logic scheduler_busy, scheduler_done, scheduler_error, pack_busy, pack_done, pack_error;
    logic b_pack_busy, b_pack_done, b_pack_error;
    logic elementwise_busy, elementwise_done, elementwise_error;
    logic rowwise_busy,rowwise_done,rowwise_error;
    logic accelerator_busy, accelerator_done, accelerator_error;
    logic done_latched, error_latched, pipeline_enable;
    logic op_start, op_done, scheduler_memory_stall, input_stall, output_stall;

    logic g_read_req_valid, g_read_req_ready, g_read_req_wide, g_read_req_qword, g_read_req_allow_ahead;
    logic [31:0] g_read_req_addr; logic [6:0] g_read_req_len;
    logic g_read_qword_ready, g_write_req_valid, g_write_req_ready, g_write_req_last;
    logic [31:0] g_write_req_addr, g_write_req_data; logic [3:0] g_write_req_strb;
    logic g_op_start, g_op_done, g_memory_stall, g_input_stall, g_output_stall;
    logic q_read_req_valid, q_read_req_ready; logic [31:0] q_read_req_addr; logic [4:0] q_read_req_len;
    logic q_read_word_ready, q_write_req_valid, q_write_req_ready;
    logic [31:0] q_write_req_addr, q_write_req_data; logic [3:0] q_write_req_strb;
    logic q_busy, q_done, q_error, q_memory_stall, q_input_stall, q_output_stall;
    logic p_read_req_valid, p_read_req_ready, p_read_req_wide, p_read_req_qword, p_read_req_allow_ahead;
    logic [31:0] p_read_req_addr; logic [6:0] p_read_req_len;
    logic p_read_word_ready, p_write_req_valid, p_write_req_ready, p_write_req_last;
    logic [31:0] p_write_req_addr, p_write_req_data; logic [3:0] p_write_req_strb;
    logic p_op_start, p_op_done, p_memory_stall, p_input_stall, p_output_stall;
    logic t_read_req_valid, t_read_req_ready, t_read_req_wide, t_read_req_qword, t_read_req_allow_ahead;
    logic [31:0] t_read_req_addr; logic [6:0] t_read_req_len;
    logic t_read_data_ready, t_write_req_valid, t_write_req_ready, t_write_req_last;
    logic [31:0] t_write_req_addr, t_write_req_data; logic [3:0] t_write_req_strb;
    logic t_op_start, t_op_done, t_memory_stall, t_input_stall, t_output_stall;
    logic e_read_req_valid,e_read_req_ready,e_read_req_wide,e_read_req_qword,e_read_req_allow_ahead;
    logic[31:0]e_read_req_addr;logic[6:0]e_read_req_len;logic e_read_qword_ready;
    logic e_write_req_valid,e_write_req_ready,e_write_req_last;logic[31:0]e_write_req_addr,e_write_req_data;logic[3:0]e_write_req_strb;
    logic e_op_start,e_op_done,e_memory_stall,e_input_stall,e_output_stall;
    logic r_read_req_valid,r_read_req_ready,r_read_req_wide,r_read_req_qword,r_read_req_allow_ahead,r_read_data_ready;
    logic[31:0]r_read_req_addr;logic[6:0]r_read_req_len;logic r_write_req_valid,r_write_req_ready,r_write_req_last;logic[31:0]r_write_req_addr,r_write_req_data;logic[3:0]r_write_req_strb;logic r_op_start,r_op_done,r_memory_stall,r_input_stall,r_output_stall;
    logic h_read_req_valid,h_read_req_ready,h_read_req_wide,h_read_req_qword,h_read_req_allow_ahead,h_read_data_ready,h_read_qword_ready;
    logic[31:0]h_read_req_addr;logic[6:0]h_read_req_len;logic h_write_req_valid,h_write_req_ready,h_write_req_last;logic[31:0]h_write_req_addr,h_write_req_data;logic[3:0]h_write_req_strb;logic h_busy,h_done,h_error,h_op_start,h_op_done,h_memory_stall,h_input_stall,h_output_stall;

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
    wire [31:0] a_stride = regs_flat[5*32 +: 32];
    wire [31:0] b_stride = regs_flat[6*32 +: 32];
    wire [15:0] dim_m = regs_flat[8*32 +: 16];
    wire [15:0] dim_n = regs_flat[9*32 +: 16];
    wire [15:0] dim_k = regs_flat[10*32 +: 16];
    wire signed [31:0] rq_multiplier = regs_flat[13*32 +: 32];
    wire [5:0] rq_right_shift = regs_flat[14*32 +: 6];
    wire signed [31:0] rq_output_zero_point = regs_flat[15*32 +: 32];

    assign start_pulse = ctrl_word[0] && !start_d;
    assign accelerator_busy=scheduler_busy||q_busy||pack_busy||b_pack_busy||elementwise_busy||rowwise_busy||h_busy;
    assign accelerator_done=scheduler_done||q_done||pack_done||b_pack_done||elementwise_done||rowwise_done||h_done;
    assign accelerator_error=scheduler_error||q_error||pack_error||b_pack_error||elementwise_error||rowwise_error||h_error;
    assign status_word = {24'd0, memory_error_code, memory_error, 1'b0, error_latched, done_latched, accelerator_busy};
    assign op_start=h_op_start||r_op_start||e_op_start||t_op_start||p_op_start||g_op_start||(start_pulse&&ctrl_word[1]);
    assign op_done=h_op_done||r_op_done||e_op_done||t_op_done||p_op_done||g_op_done||q_done;
    assign q_memory_stall=q_busy&&memory_busy;
    assign q_input_stall=q_read_req_valid&&!q_read_req_ready;
    assign q_output_stall=q_write_req_valid&&!q_write_req_ready;
    assign scheduler_memory_stall=active_prune?h_memory_stall:(active_rowwise?r_memory_stall:(active_elementwise?e_memory_stall:(active_b_pack?t_memory_stall:(active_pack?p_memory_stall:(active_requant?q_memory_stall:g_memory_stall)))));
    assign input_stall=active_prune?h_input_stall:(active_rowwise?r_input_stall:(active_elementwise?e_input_stall:(active_b_pack?t_input_stall:(active_pack?p_input_stall:(active_requant?q_input_stall:g_input_stall)))));
    assign output_stall=active_prune?h_output_stall:(active_rowwise?r_output_stall:(active_elementwise?e_output_stall:(active_b_pack?t_output_stall:(active_pack?p_output_stall:(active_requant?q_output_stall:g_output_stall)))));
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
        if(!aresetn) begin start_d<=0;active_requant<=0;active_pack<=0;active_b_pack<=0;active_elementwise<=0;active_rowwise<=0;active_prune<=0;done_latched<=0;error_latched<=0; end
        else begin
            start_d <= ctrl_word[0];
            if(start_pulse) begin active_requant<=ctrl_word[1];active_pack<=ctrl_word[12];active_b_pack<=ctrl_word[13];active_elementwise<=ctrl_word[16]||ctrl_word[17];active_rowwise<=ctrl_word[18]||ctrl_word[19];active_prune<=ctrl_word[23];done_latched<=0;error_latched<=0; end
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
        .op_code((active_prune||h_op_start)?8'd7:((active_rowwise||r_op_start)?8'd5:((active_elementwise||e_op_start)?8'd4:((active_b_pack||t_op_start)?8'd3:((active_pack||p_op_start)?8'd2:(active_requant?8'd6:8'd1)))))),
        .error_event(accelerator_error||memory_error), .debug_data({28'd0,memory_error,memory_error_code}),
        .pipeline_enable, .debug_regs_flat
    );

    transformer_gemm_scheduler scheduler (
        .clk(aclk), .rst_n(aresetn), .pipeline_enable, .start(start_pulse && !ctrl_word[1] && !(|ctrl_word[19:12]) && !ctrl_word[23]),
        .weight_int4(ctrl_word[11]), .fused_requant(ctrl_word[20]),
        .fused_gelu(ctrl_word[24]), .fused_output_mode(ctrl_word[22:21]),
        .a_base, .b_base, .c_base, .c_stride, .bias_base(b_stride),
        .rq_multiplier, .rq_right_shift, .rq_output_zero_point,
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
        .memory_busy, .memory_error(memory_error&&!active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise)
    );

    generate if (ENABLE_LEGACY_STANDALONE) begin : gen_legacy_standalone
    requantize_ddr requantizer (
        .clk(aclk), .rst_n(aresetn), .start(start_pulse && ctrl_word[1]),
        .input_base_addr(a_base), .output_base_addr(c_base), .element_count({16'd0, dim_m}),
        .multiplier(rq_multiplier), .right_shift(rq_right_shift), .output_zero_point(rq_output_zero_point),
        .busy(q_busy), .done(q_done), .error(q_error),
        .read_req_valid(q_read_req_valid), .read_req_ready(q_read_req_ready),
        .read_req_addr(q_read_req_addr), .read_req_len(q_read_req_len),
        .read_word_valid(read_word_valid && active_requant), .read_word_ready(q_read_word_ready),
        .read_word_data(read_word_data), .read_word_keep(read_word_keep), .read_word_last(read_word_last),
        .write_req_valid(q_write_req_valid), .write_req_ready(q_write_req_ready),
        .write_req_addr(q_write_req_addr), .write_req_data(q_write_req_data), .write_req_strb(q_write_req_strb)
    );

    requantize_pack_int8 packer (
        .clk(aclk), .rst_n(aresetn), .pipeline_enable, .start(start_pulse && ctrl_word[12] && !ctrl_word[13]),
        .input_base_addr(a_base), .output_base_addr(c_base), .input_row_stride(a_stride),
        .rows(dim_m), .columns(dim_n), .multiplier(rq_multiplier), .right_shift(rq_right_shift),
        .output_zero_point(rq_output_zero_point), .busy(pack_busy), .done(pack_done), .error(pack_error),
        .op_start(p_op_start), .op_done(p_op_done), .memory_stall(p_memory_stall),
        .input_stall(p_input_stall), .output_stall(p_output_stall),
        .read_req_valid(p_read_req_valid), .read_req_ready(p_read_req_ready), .read_req_addr(p_read_req_addr),
        .read_req_len(p_read_req_len), .read_req_wide(p_read_req_wide), .read_req_qword(p_read_req_qword),
        .read_req_allow_ahead(p_read_req_allow_ahead), .read_word_valid, .read_word_ready(p_read_word_ready),
        .read_word_data, .read_word_keep, .read_word_last,
        .write_req_valid(p_write_req_valid), .write_req_ready(p_write_req_ready), .write_req_addr(p_write_req_addr),
        .write_req_data(p_write_req_data), .write_req_strb(p_write_req_strb), .write_req_last(p_write_req_last),
        .memory_busy, .memory_error(memory_error && active_pack && !active_b_pack)
    );

    activation_b_pack_int8 b_packer (
        .clk(aclk), .rst_n(aresetn), .pipeline_enable, .start(start_pulse && ctrl_word[13]),
        .transpose(ctrl_word[14]), .source_row_major(ctrl_word[15]),
        .input_base_addr(a_base), .output_base_addr(c_base),
        .reduction(dim_m), .outputs(dim_n), .busy(b_pack_busy), .done(b_pack_done), .error(b_pack_error),
        .op_start(t_op_start), .op_done(t_op_done), .memory_stall(t_memory_stall),
        .input_stall(t_input_stall), .output_stall(t_output_stall),
        .read_req_valid(t_read_req_valid), .read_req_ready(t_read_req_ready), .read_req_addr(t_read_req_addr),
        .read_req_len(t_read_req_len), .read_req_wide(t_read_req_wide), .read_req_qword(t_read_req_qword),
        .read_req_allow_ahead(t_read_req_allow_ahead), .read_data_valid, .read_data_ready(t_read_data_ready),
        .read_data, .read_data_last,
        .write_req_valid(t_write_req_valid), .write_req_ready(t_write_req_ready), .write_req_addr(t_write_req_addr),
        .write_req_data(t_write_req_data), .write_req_strb(t_write_req_strb), .write_req_last(t_write_req_last),
        .memory_busy, .memory_error(memory_error && active_b_pack)
    );
    end else begin : gen_no_legacy_standalone
        // The resident DeiT graph fuses requantization and all activation
        // layout transforms into producer operators.  Keep the legacy engines
        // available to simulation/source users, but omit them from the
        // production image to recover routing and timing margin.
        assign q_busy = 1'b0; assign q_done = 1'b0; assign q_error = 1'b0;
        assign q_read_req_valid = 1'b0; assign q_read_req_addr = '0; assign q_read_req_len = '0;
        assign q_read_word_ready = 1'b0; assign q_write_req_valid = 1'b0;
        assign q_write_req_addr = '0; assign q_write_req_data = '0; assign q_write_req_strb = '0;

        assign pack_busy = 1'b0; assign pack_done = 1'b0; assign pack_error = 1'b0;
        assign p_op_start = 1'b0; assign p_op_done = 1'b0; assign p_memory_stall = 1'b0;
        assign p_input_stall = 1'b0; assign p_output_stall = 1'b0;
        assign p_read_req_valid = 1'b0; assign p_read_req_addr = '0; assign p_read_req_len = '0;
        assign p_read_req_wide = 1'b0; assign p_read_req_qword = 1'b0;
        assign p_read_req_allow_ahead = 1'b0; assign p_read_word_ready = 1'b0;
        assign p_write_req_valid = 1'b0; assign p_write_req_addr = '0;
        assign p_write_req_data = '0; assign p_write_req_strb = '0; assign p_write_req_last = 1'b0;

        assign b_pack_busy = 1'b0; assign b_pack_done = 1'b0; assign b_pack_error = 1'b0;
        assign t_op_start = 1'b0; assign t_op_done = 1'b0; assign t_memory_stall = 1'b0;
        assign t_input_stall = 1'b0; assign t_output_stall = 1'b0;
        assign t_read_req_valid = 1'b0; assign t_read_req_addr = '0; assign t_read_req_len = '0;
        assign t_read_req_wide = 1'b0; assign t_read_req_qword = 1'b0;
        assign t_read_req_allow_ahead = 1'b0; assign t_read_data_ready = 1'b0;
        assign t_write_req_valid = 1'b0; assign t_write_req_addr = '0;
        assign t_write_req_data = '0; assign t_write_req_strb = '0; assign t_write_req_last = 1'b0;
    end endgenerate

    transformer_elementwise_ddr elementwise (
        .clk(aclk),.rst_n(aresetn),.pipeline_enable,.start(start_pulse&&(ctrl_word[16]||ctrl_word[17])),.residual_mode(ctrl_word[17]),
        .input0_base_addr(a_base),.input1_base_addr(b_base),.output_base_addr(c_base),.total_bytes(dim_m*dim_n),
        .lhs_right_shift(rq_right_shift[3:0]),.rhs_right_shift(rq_output_zero_point[3:0]),
        .busy(elementwise_busy),.done(elementwise_done),.error(elementwise_error),.op_start(e_op_start),.op_done(e_op_done),
        .memory_stall(e_memory_stall),.input_stall(e_input_stall),.output_stall(e_output_stall),
        .read_req_valid(e_read_req_valid),.read_req_ready(e_read_req_ready),.read_req_addr(e_read_req_addr),.read_req_len(e_read_req_len),
        .read_req_wide(e_read_req_wide),.read_req_qword(e_read_req_qword),.read_req_allow_ahead(e_read_req_allow_ahead),
        .read_qword_valid,.read_qword_ready(e_read_qword_ready),.read_qword_data,.read_qword_keep,.read_qword_last,
        .write_req_valid(e_write_req_valid),.write_req_ready(e_write_req_ready),.write_req_addr(e_write_req_addr),.write_req_data(e_write_req_data),
        .write_req_strb(e_write_req_strb),.write_req_last(e_write_req_last),.memory_busy,.memory_error(memory_error&&active_elementwise));

    transformer_rowwise_ddr rowwise(.clk(aclk),.rst_n(aresetn),.pipeline_enable,.start(start_pulse&&(ctrl_word[18]||ctrl_word[19])),.softmax_mode(ctrl_word[19]),.output_tiled(ctrl_word[18]||ctrl_word[21]),
        .residual_mode(ctrl_word[25]),.input_base_addr(a_base),.residual_base_addr(a_stride),
        .lhs_right_shift(rq_right_shift[3:0]),.rhs_right_shift(rq_output_zero_point[3:0]),
        .gamma_base_addr(b_base),.beta_base_addr(b_stride),.output_base_addr(c_base),.epsilon_term(rq_multiplier),.rows(dim_m),.columns(dim_n),
        .busy(rowwise_busy),.done(rowwise_done),.error(rowwise_error),.op_start(r_op_start),.op_done(r_op_done),.memory_stall(r_memory_stall),.input_stall(r_input_stall),.output_stall(r_output_stall),
        .read_req_valid(r_read_req_valid),.read_req_ready(r_read_req_ready),.read_req_addr(r_read_req_addr),.read_req_len(r_read_req_len),.read_req_wide(r_read_req_wide),.read_req_qword(r_read_req_qword),.read_req_allow_ahead(r_read_req_allow_ahead),
        .read_data_valid,.read_data_ready(r_read_data_ready),.read_data,.read_data_last,.write_req_valid(r_write_req_valid),.write_req_ready(r_write_req_ready),.write_req_addr(r_write_req_addr),.write_req_data(r_write_req_data),.write_req_strb(r_write_req_strb),.write_req_last(r_write_req_last),.memory_busy,.memory_error(memory_error&&active_rowwise));

    heatvit_token_pruner_ddr heatvit_pruner(.clk(aclk),.rst_n(aresetn),.pipeline_enable,.start(start_pulse&&ctrl_word[23]),.weight_int4(ctrl_word[11]),
        .input_base_addr(a_base),.weight_base_addr(b_base),.output_base_addr(c_base),.input_tokens(dim_m),.output_tokens(dim_n),.features(dim_k),.package_reciprocal_q16(rq_multiplier),
        .busy(h_busy),.done(h_done),.error(h_error),.op_start(h_op_start),.op_done(h_op_done),.memory_stall(h_memory_stall),.input_stall(h_input_stall),.output_stall(h_output_stall),
        .read_req_valid(h_read_req_valid),.read_req_ready(h_read_req_ready),.read_req_addr(h_read_req_addr),.read_req_len(h_read_req_len),.read_req_wide(h_read_req_wide),.read_req_qword(h_read_req_qword),.read_req_allow_ahead(h_read_req_allow_ahead),
        .read_data_valid,.read_data_ready(h_read_data_ready),.read_data,.read_data_last,.read_qword_valid,.read_qword_ready(h_read_qword_ready),.read_qword_data,.read_qword_keep,.read_qword_last,
        .write_req_valid(h_write_req_valid),.write_req_ready(h_write_req_ready),.write_req_addr(h_write_req_addr),.write_req_data(h_write_req_data),.write_req_strb(h_write_req_strb),.write_req_last(h_write_req_last),.memory_busy,.memory_error(memory_error&&active_prune));

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
    always_comb begin
        arb_read_req_valid=active_prune?h_read_req_valid:(active_rowwise?r_read_req_valid:(active_elementwise?e_read_req_valid:(active_b_pack?t_read_req_valid:(active_pack?p_read_req_valid:(active_requant?q_read_req_valid:g_read_req_valid)))));
        arb_read_req_addr=active_prune?h_read_req_addr:(active_rowwise?r_read_req_addr:(active_elementwise?e_read_req_addr:(active_b_pack?t_read_req_addr:(active_pack?p_read_req_addr:(active_requant?q_read_req_addr:g_read_req_addr)))));
        arb_read_req_len=active_prune?h_read_req_len:(active_rowwise?r_read_req_len:(active_elementwise?e_read_req_len:(active_b_pack?t_read_req_len:(active_pack?p_read_req_len:(active_requant?{2'd0,q_read_req_len}:g_read_req_len)))));
        arb_read_req_wide=active_prune?h_read_req_wide:(active_rowwise?r_read_req_wide:(active_elementwise?e_read_req_wide:(active_b_pack?t_read_req_wide:(active_pack?p_read_req_wide:(active_requant?1'b1:g_read_req_wide)))));
        arb_read_req_qword=active_prune?h_read_req_qword:(active_rowwise?r_read_req_qword:(active_elementwise?e_read_req_qword:(active_b_pack?t_read_req_qword:(active_pack?p_read_req_qword:(active_requant?1'b0:g_read_req_qword)))));
        arb_read_req_allow_ahead=active_prune?h_read_req_allow_ahead:(active_rowwise?r_read_req_allow_ahead:(active_elementwise?e_read_req_allow_ahead:(active_b_pack?t_read_req_allow_ahead:(active_pack?p_read_req_allow_ahead:(active_requant?1'b0:g_read_req_allow_ahead)))));
        g_read_req_ready=!active_requant&&!active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&read_req_pipe_ready;q_read_req_ready=active_requant&&!active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&read_req_pipe_ready;p_read_req_ready=active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&read_req_pipe_ready;t_read_req_ready=active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&read_req_pipe_ready;e_read_req_ready=active_elementwise&&!active_rowwise&&!active_prune&&read_req_pipe_ready;r_read_req_ready=active_rowwise&&!active_prune&&read_req_pipe_ready;h_read_req_ready=active_prune&&read_req_pipe_ready;
        read_qword_ready=active_prune?h_read_qword_ready:(active_elementwise?e_read_qword_ready:(!active_pack&&!active_b_pack&&!active_rowwise&&g_read_qword_ready));
        read_word_ready = active_requant ? q_read_word_ready : (active_pack && !active_b_pack && p_read_word_ready);
        read_data_ready=active_prune?h_read_data_ready:(active_rowwise?r_read_data_ready:(active_b_pack?t_read_data_ready:1'b1));
        arb_write_req_valid=active_prune?h_write_req_valid:(active_rowwise?r_write_req_valid:(active_elementwise?e_write_req_valid:(active_b_pack?t_write_req_valid:(active_pack?p_write_req_valid:(active_requant?q_write_req_valid:g_write_req_valid)))));
        arb_write_req_addr=active_prune?h_write_req_addr:(active_rowwise?r_write_req_addr:(active_elementwise?e_write_req_addr:(active_b_pack?t_write_req_addr:(active_pack?p_write_req_addr:(active_requant?q_write_req_addr:g_write_req_addr)))));
        arb_write_req_data=active_prune?h_write_req_data:(active_rowwise?r_write_req_data:(active_elementwise?e_write_req_data:(active_b_pack?t_write_req_data:(active_pack?p_write_req_data:(active_requant?q_write_req_data:g_write_req_data)))));
        arb_write_req_strb=active_prune?h_write_req_strb:(active_rowwise?r_write_req_strb:(active_elementwise?e_write_req_strb:(active_b_pack?t_write_req_strb:(active_pack?p_write_req_strb:(active_requant?q_write_req_strb:g_write_req_strb)))));
        arb_write_req_last=active_prune?h_write_req_last:(active_rowwise?r_write_req_last:(active_elementwise?e_write_req_last:(active_b_pack?t_write_req_last:(active_pack?p_write_req_last:(active_requant?1'b1:g_write_req_last)))));
        g_write_req_ready=!active_requant&&!active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&write_req_pipe_ready;q_write_req_ready=active_requant&&!active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&write_req_pipe_ready;p_write_req_ready=active_pack&&!active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&write_req_pipe_ready;t_write_req_ready=active_b_pack&&!active_elementwise&&!active_rowwise&&!active_prune&&write_req_pipe_ready;e_write_req_ready=active_elementwise&&!active_rowwise&&!active_prune&&write_req_pipe_ready;r_write_req_ready=active_rowwise&&!active_prune&&write_req_pipe_ready;h_write_req_ready=active_prune&&write_req_pipe_ready;
    end
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
