module bcuda_tiled_systolic_ddr_top #(
    parameter int MAX_ARRAYS = 1
) (
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET aresetn" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    input  logic        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    input  logic        aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  logic [5:0]  s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  logic        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  logic [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  logic [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  logic        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  logic        s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  logic [5:0]  s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  logic        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  logic        s_axi_rready,

    output logic        mem_cmd_valid,
    input  logic        mem_cmd_ready,
    output logic        mem_cmd_write,
    output logic [31:0] mem_cmd_addr,
    output logic [3:0]  mem_cmd_len,
    output logic [31:0] mem_cmd_wdata,
    output logic [3:0]  mem_cmd_wstrb,
    output logic        mem_write_valid,
    input  logic        mem_write_ready,
    output logic [31:0] mem_write_data,
    output logic [3:0]  mem_write_strb,
    output logic        mem_write_last,

    input  logic        mem_read_valid,
    output logic        mem_read_ready,
    input  logic [31:0] mem_read_data,
    input  logic        mem_read_last
);

    typedef enum logic [3:0] {
        OP_IDLE,
        OP_ZERO_A,
        OP_LOAD_A,
        OP_ZERO_B,
        OP_LOAD_B,
        OP_LOAD_B_NEXT,
        OP_BIAS_PRELOAD,
        OP_COMPUTE_WAIT,
        OP_ACCUMULATE,
        OP_WRITE_C
    } op_state_t;

    op_state_t op_state;

    logic [31:0] ctrl_word;
    logic [31:0] status_word;
    logic [32*16-1:0] regs_flat;
    logic ctrl_start_d;
    logic start_pulse;
    logic done_r;
    logic error_r;
    logic write_complete_pending;

    wire [31:0] a_base = regs_flat[2*32 +: 32];
    wire [31:0] b_base = regs_flat[3*32 +: 32];
    wire [31:0] c_base = regs_flat[4*32 +: 32];
    wire [31:0] a_stride = regs_flat[5*32 +: 32];
    wire [31:0] b_stride = regs_flat[6*32 +: 32];
    wire [31:0] c_stride = regs_flat[7*32 +: 32];
    wire [15:0] dim_m = regs_flat[8*32 +: 16];
    wire [15:0] dim_n = regs_flat[9*32 +: 16];
    wire [15:0] dim_k = regs_flat[10*32 +: 16];
    wire [4:0] tile_size = regs_flat[11*32 +: 5];
    wire [7:0] array_count = regs_flat[12*32 +: 8];
    wire signed [31:0] rq_multiplier = regs_flat[13*32 +: 32];
    wire [31:0] rq_right_shift = regs_flat[14*32 +: 32];
    wire signed [31:0] rq_output_zero_point = regs_flat[15*32 +: 32];
    wire [31:0] conv_bias_base = regs_flat[15*32 +: 32];
    wire requantize_mode = ctrl_word[1];
    wire conv2d_mode = ctrl_word[2];
    wire activation_mode = ctrl_word[3];
    wire activation_leaky_mode = ctrl_word[4];
    // In CONV2D mode the otherwise GEMM-specific geometry registers encode:
    // A_STRIDE=C, B_STRIDE=H, C_STRIDE=W, M=out_C, N=out_H, K=out_W,
    // TILE_SIZE=kernel_H, ARRAY_COUNT=kernel_W, RQ_MULTIPLIER=strides,
    // RQ_RIGHT_SHIFT=pads. Batch is deliberately fixed to one.
    wire [15:0] conv_input_c = a_stride[15:0];
    wire [15:0] conv_input_h = b_stride[15:0];
    wire [15:0] conv_input_w = c_stride[15:0];
    wire [15:0] conv_out_c = dim_m;
    wire [15:0] conv_out_h = dim_n;
    wire [15:0] conv_out_w = dim_k;
    wire [15:0] conv_gemm_m = conv_out_h * conv_out_w;
    wire [15:0] conv_gemm_k = conv_input_c * tile_size * array_count;
    wire [7:0] conv_stride_h = rq_multiplier[7:0];
    wire [7:0] conv_stride_w = rq_multiplier[15:8];
    wire [7:0] conv_pad_h = rq_right_shift[7:0];
    wire [7:0] conv_pad_w = rq_right_shift[15:8];

    logic controller_start;
    logic controller_busy;
    logic controller_done;
    logic controller_error;
    logic load_a_start;
    logic load_a_done;
    logic load_b_start;
    logic load_b_done;
    logic compute_start;
    logic compute_done;
    logic write_c_start;
    logic write_c_done;
    logic [15:0] m0;
    logic [15:0] n0;
    logic [15:0] k0;
    logic [4:0] m_len;
    logic [15:0] n_group_cols;
    logic [4:0] k_len;
    logic [7:0] active_arrays;
    logic clear_accumulators;

    logic loader_start;
    logic loader_matrix_select;
    logic [31:0] loader_base_addr;
    logic [31:0] loader_row_stride;
    logic [4:0] loader_rows;
    logic [4:0] loader_cols;
    logic loader_busy;
    logic loader_done;
    logic loader_error;
    logic loader_read_req_valid;
    logic loader_read_req_ready;
    logic [31:0] loader_read_req_addr;
    logic loader_read_data_valid;
    logic loader_read_data_ready;
    logic [7:0] loader_read_data;
    logic loader_cache_we;
    logic loader_cache_matrix;
    logic [7:0] loader_cache_addr;
    logic [7:0] loader_cache_wdata;

    logic core_start;
    logic core_busy;
    logic core_done;
    logic [2:0] core_result_array;
    logic [7:0] core_result_addr;
    logic [31:0] core_result_data;

    logic [7:0] zero_idx;
    logic [7:0] accum_idx;
    logic [2:0] zero_b_array_idx;
    logic [2:0] load_b_array_idx;
    logic [2:0] accum_array_idx;
    logic [2:0] write_array_idx;
    logic accum_op_valid;
    logic accum_op_ready;
    logic accum_op_clear;
    logic [31:0] accum_op_result;
    logic [7:0] accum_read_addr;
    logic accum_read_req;
    logic [MAX_ARRAYS-1:0] accum_read_valid_array;
    logic signed [31:0] accum_read_data [0:MAX_ARRAYS-1];
    logic [MAX_ARRAYS-1:0] accum_op_ready_array;

    logic writer_start;
    logic writer_busy;
    logic writer_done;
    logic writer_error;
    logic [7:0] writer_tile_read_addr;
    logic writer_tile_read_req;
    logic writer_tile_read_valid;
    logic [31:0] writer_tile_read_data;
    logic writer_write_req_valid;
    logic writer_write_req_ready;
    logic [31:0] writer_write_req_addr;
    logic [31:0] writer_write_req_data;
    logic [3:0] writer_write_req_strb;
    logic writer_write_req_last;

    logic rq_start;
    logic rq_busy;
    logic rq_done;
    logic rq_error;
    logic rq_read_req_valid;
    logic rq_read_req_ready;
    logic [31:0] rq_read_req_addr;
    logic rq_read_data_valid;
    logic rq_read_data_ready;
    logic [7:0] rq_read_data;
    logic rq_write_req_valid;
    logic rq_write_req_ready;
    logic [31:0] rq_write_req_addr;
    logic [31:0] rq_write_req_data;
    logic [3:0] rq_write_req_strb;

    logic bias_preload_start, bias_preload_busy, bias_preload_done, bias_preload_error;
    logic bias_read_req_valid, bias_read_req_ready, bias_read_data_valid, bias_read_data_ready;
    logic [31:0] bias_read_req_addr;
    logic [7:0] bias_read_data;
    logic bias_accum_valid;
    logic [7:0] bias_accum_addr;
    logic signed [31:0] bias_accum_data;

    logic activation_start;
    logic activation_busy, activation_done, activation_error;
    logic activation_read_req_valid, activation_read_req_ready;
    logic [31:0] activation_read_req_addr;
    logic activation_read_data_valid, activation_read_data_ready;
    logic [7:0] activation_read_data;
    logic activation_write_req_valid, activation_write_req_ready;
    logic [31:0] activation_write_req_addr, activation_write_req_data;
    logic [3:0] activation_write_req_strb;

    logic memio_read_req_valid;
    logic memio_read_req_ready;
    logic [31:0] memio_read_req_addr;
    logic memio_read_data_valid;
    logic memio_read_data_ready;
    logic [7:0] memio_read_data;
    logic memio_busy;
    logic memio_error;
    logic memio_write_req_ready;

    wire [4:0] b_len = (n_group_cols > (load_b_array_idx * 16 + 16)) ? 5'd16 : (n_group_cols - load_b_array_idx * 16);
    wire [4:0] writer_n_len = (n_group_cols > (write_array_idx * 16 + 16)) ? 5'd16 : (n_group_cols - write_array_idx * 16);

    assign start_pulse = ctrl_word[0] && !ctrl_start_d;
    assign controller_start = start_pulse && !requantize_mode && !activation_mode;
    assign rq_start = start_pulse && requantize_mode;
    assign activation_start = start_pulse && activation_mode;
    // Bits 2:0 retain the firmware ABI. Upper bits expose live board-level
    // progress so an HP0 handshake stall can be diagnosed through AXI-Lite.
    assign status_word = {
        14'd0,
        loader_read_req_valid, loader_read_req_ready,
        loader_read_data_valid, loader_read_data_ready,
        mem_cmd_ready, mem_cmd_valid, memio_busy,
        writer_busy, core_busy, loader_busy, controller_busy,
        op_state,
        error_r, done_r,
        controller_busy || rq_busy || activation_busy || op_state != OP_IDLE
    };

    assign load_a_done = (op_state == OP_LOAD_A) && loader_done;
    assign load_b_done = (op_state == OP_LOAD_B) && loader_done;
    assign compute_done = (op_state == OP_ACCUMULATE) && accum_op_valid && accum_op_ready && accum_idx == 8'hff && accum_array_idx == active_arrays - 1'b1;
    assign write_c_done = (op_state == OP_WRITE_C) && write_complete_pending && !memio_busy;

    assign loader_start = ((op_state == OP_ZERO_A) && zero_idx == 8'hff) ||
                          (op_state == OP_LOAD_B_NEXT);
    assign loader_matrix_select = (op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT);
    assign loader_base_addr = loader_matrix_select ? (b_base + (k0 * b_stride) + n0 + load_b_array_idx * 16) :
                                                     (a_base + (m0 * a_stride) + k0);
    assign loader_row_stride = loader_matrix_select ? b_stride : a_stride;
    assign loader_rows = loader_matrix_select ? k_len : m_len;
    assign loader_cols = loader_matrix_select ? b_len : k_len;

    assign memio_read_req_valid = bias_preload_busy ? bias_read_req_valid : (activation_busy ? activation_read_req_valid : (rq_busy ? rq_read_req_valid : loader_read_req_valid));
    assign loader_read_req_ready = !bias_preload_busy && !activation_busy && !rq_busy && memio_read_req_ready;
    assign rq_read_req_ready = !bias_preload_busy && !activation_busy && rq_busy && memio_read_req_ready;
    assign activation_read_req_ready = !bias_preload_busy && activation_busy && memio_read_req_ready;
    assign bias_read_req_ready = bias_preload_busy && memio_read_req_ready;
    assign memio_read_req_addr = bias_preload_busy ? bias_read_req_addr : (activation_busy ? activation_read_req_addr : (rq_busy ? rq_read_req_addr : loader_read_req_addr));
    assign loader_read_data_valid = !bias_preload_busy && !activation_busy && !rq_busy && memio_read_data_valid;
    assign rq_read_data_valid = !bias_preload_busy && !activation_busy && rq_busy && memio_read_data_valid;
    assign activation_read_data_valid = !bias_preload_busy && activation_busy && memio_read_data_valid;
    assign bias_read_data_valid = bias_preload_busy && memio_read_data_valid;
    assign memio_read_data_ready = bias_preload_busy ? bias_read_data_ready : (activation_busy ? activation_read_data_ready : (rq_busy ? rq_read_data_ready : loader_read_data_ready));
    assign loader_read_data = memio_read_data;
    assign rq_read_data = memio_read_data;
    assign activation_read_data = memio_read_data;
    assign bias_read_data = memio_read_data;

    assign accum_op_valid = (op_state == OP_ACCUMULATE);
    // A biased Conv2D tile has already been initialized by the preload engine;
    // its first systolic pass must add, not clear that seeded accumulator state.
    assign accum_op_clear = (op_state == OP_BIAS_PRELOAD) ? 1'b1 :
                            ((conv2d_mode && conv_bias_base != 0) ? 1'b0 : clear_accumulators);
    assign core_result_addr = (op_state == OP_BIAS_PRELOAD) ? bias_accum_addr : accum_idx;
    assign core_result_array = accum_array_idx;
    assign accum_read_addr = writer_tile_read_addr;
    assign accum_read_req = writer_tile_read_req;
    assign writer_tile_read_data = accum_read_data[write_array_idx];
    assign writer_tile_read_valid = accum_read_valid_array[write_array_idx];

    assign writer_start = write_c_start || (op_state == OP_WRITE_C && writer_done && write_array_idx < active_arrays - 1'b1);
    assign writer_write_req_strb = 4'hf;
    axi_lite_control control (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .status_word(status_word),
        .ctrl_word(ctrl_word),
        .regs_flat(regs_flat)
    );

    gemm_controller #(
        .MAX_ARRAYS(MAX_ARRAYS)
    ) controller (
        .clk(aclk),
        .rst_n(aresetn),
        .start(controller_start),
        .M(conv2d_mode ? conv_gemm_m : dim_m),
        .N(conv2d_mode ? conv_out_c : dim_n),
        .K(conv2d_mode ? conv_gemm_k : dim_k),
        .tile_size(conv2d_mode ? 5'd16 : tile_size),
        .array_count(conv2d_mode ? 8'd1 : array_count),
        .busy(controller_busy),
        .done(controller_done),
        .error(controller_error),
        .load_a_start(load_a_start),
        .load_a_done(load_a_done),
        .load_b_start(load_b_start),
        .load_b_done(load_b_done),
        .compute_start(compute_start),
        .compute_done(compute_done),
        .write_c_start(write_c_start),
        .write_c_done(write_c_done),
        .m_tile(),
        .n_group_tile(),
        .k_tile(),
        .m0(m0),
        .n0(n0),
        .k0(k0),
        .m_len(m_len),
        .n_group_cols(n_group_cols),
        .k_len(k_len),
        .active_arrays(active_arrays),
        .clear_accumulators(clear_accumulators),
        .write_c_tiles()
    );

    memory_io_driver memio (
        .clk(aclk),
        .rst_n(aresetn),
        .read_req_valid(memio_read_req_valid),
        .read_req_ready(memio_read_req_ready),
        .read_req_addr(memio_read_req_addr),
        .read_data_valid(memio_read_data_valid),
        .read_data_ready(memio_read_data_ready),
        .read_data(memio_read_data),
        .write_req_valid(activation_busy ? activation_write_req_valid : (rq_busy ? rq_write_req_valid : writer_write_req_valid)),
        .write_req_ready(memio_write_req_ready),
        .write_req_addr(activation_busy ? activation_write_req_addr : (rq_busy ? rq_write_req_addr : writer_write_req_addr)),
        .write_req_data(activation_busy ? activation_write_req_data : (rq_busy ? rq_write_req_data : writer_write_req_data)),
        .write_req_strb(activation_busy ? activation_write_req_strb : (rq_busy ? rq_write_req_strb : writer_write_req_strb)),
        .write_req_last(activation_busy ? 1'b1 : (rq_busy ? 1'b1 : writer_write_req_last)),
        .mem_cmd_valid(mem_cmd_valid),
        .mem_cmd_ready(mem_cmd_ready),
        .mem_cmd_write(mem_cmd_write),
        .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_len(mem_cmd_len),
        .mem_cmd_wdata(mem_cmd_wdata),
        .mem_cmd_wstrb(mem_cmd_wstrb),
        .mem_write_valid(mem_write_valid),
        .mem_write_ready(mem_write_ready),
        .mem_write_data(mem_write_data),
        .mem_write_strb(mem_write_strb),
        .mem_write_last(mem_write_last),
        .mem_read_valid(mem_read_valid),
        .mem_read_ready(mem_read_ready),
        .mem_read_data(mem_read_data),
        .mem_read_last(mem_read_last),
        .busy(memio_busy),
        .error(memio_error)
    );

    assign writer_write_req_ready = !activation_busy && !rq_busy && memio_write_req_ready;
    assign rq_write_req_ready = !activation_busy && rq_busy && memio_write_req_ready;
    assign activation_write_req_ready = activation_busy && memio_write_req_ready;

    conv2d_tile_loader_ddr loader (
        .clk(aclk),
        .rst_n(aresetn),
        .start(loader_start),
        .matrix_select(loader_matrix_select),
        .conv_mode(conv2d_mode),
        .base_addr(loader_base_addr),
        .row_stride(loader_row_stride),
        .input_base_addr(a_base),
        .weight_base_addr(b_base),
        .input_c(conv_input_c), .input_h(conv_input_h), .input_w(conv_input_w),
        .out_c(conv_out_c), .out_h(conv_out_h), .out_w(conv_out_w),
        .kernel_h({3'd0, tile_size}), .kernel_w(array_count),
        .stride_h(conv_stride_h), .stride_w(conv_stride_w), .pad_h(conv_pad_h), .pad_w(conv_pad_w),
        .m0(m0), .k0(k0), .n0(n0 + load_b_array_idx * 16),
        .tile_rows(loader_rows),
        .tile_cols(loader_cols),
        .busy(loader_busy),
        .done(loader_done),
        .error(loader_error),
        .read_req_valid(loader_read_req_valid),
        .read_req_ready(loader_read_req_ready),
        .read_req_addr(loader_read_req_addr),
        .read_data_valid(loader_read_data_valid),
        .read_data_ready(loader_read_data_ready),
        .read_data(loader_read_data),
        .cache_we(loader_cache_we),
        .cache_matrix(loader_cache_matrix),
        .cache_addr(loader_cache_addr),
        .cache_wdata(loader_cache_wdata)
    );

    gemm_array_cluster_8 #(
        .ARRAY_COUNT(MAX_ARRAYS)
    ) core (
        .clk(aclk),
        .rst_n(aresetn),
        .load_valid(loader_cache_we || op_state == OP_ZERO_A || op_state == OP_ZERO_B),
        .load_matrix((op_state == OP_ZERO_B) || loader_cache_matrix),
        .load_array((op_state == OP_ZERO_B) ? zero_b_array_idx : load_b_array_idx),
        .load_addr((op_state == OP_ZERO_A || op_state == OP_ZERO_B) ? zero_idx : loader_cache_addr),
        .load_data((op_state == OP_ZERO_A || op_state == OP_ZERO_B) ? 8'h00 : loader_cache_wdata),
        .start(core_start),
        .active_arrays(active_arrays[3:0]),
        .busy(core_busy),
        .done(core_done),
        .result_array(core_result_array),
        .result_addr(core_result_addr),
        .result_data(core_result_data)
    );

    generate
        for (genvar array_idx = 0; array_idx < MAX_ARRAYS; array_idx++) begin : accumulator_banks
            accumulator_buffer accumulator (
                .clk(aclk),
                .rst_n(aresetn),
                .op_valid(accum_op_valid && accum_array_idx == array_idx),
                .op_ready(accum_op_ready_array[array_idx]),
                .op_clear(accum_op_clear),
                .op_addr(accum_idx),
                .op_data((op_state == OP_BIAS_PRELOAD) ? bias_accum_data : core_result_data),
                .op_result(),
                .seed_valid(op_state == OP_BIAS_PRELOAD && bias_accum_valid && accum_array_idx == array_idx),
                .seed_ready(),
                .seed_addr(bias_accum_addr),
                .seed_data(bias_accum_data),
                .read_req(accum_read_req),
                .read_addr(accum_read_addr),
                .read_valid(accum_read_valid_array[array_idx]),
                .read_data(accum_read_data[array_idx])
            );
        end
    endgenerate
    assign accum_op_ready = accum_op_ready_array[accum_array_idx];

    assign bias_preload_start = (op_state == OP_IDLE) && compute_start && clear_accumulators && conv2d_mode && conv_bias_base != 0;
    conv_bias_preload_ddr bias_preloader (
        .clk(aclk), .rst_n(aresetn), .start(bias_preload_start), .bias_base_addr(conv_bias_base), .channel_offset(n0), .tile_rows(m_len), .tile_cols(n_group_cols[4:0]),
        .busy(bias_preload_busy), .done(bias_preload_done), .error(bias_preload_error),
        .read_req_valid(bias_read_req_valid), .read_req_ready(bias_read_req_ready), .read_req_addr(bias_read_req_addr),
        .read_data_valid(bias_read_data_valid), .read_data_ready(bias_read_data_ready), .read_data(bias_read_data),
        .accum_valid(bias_accum_valid), .accum_ready(1'b1), .accum_addr(bias_accum_addr), .accum_data(bias_accum_data)
    );

    conv2d_tile_writer_ddr writer (
        .clk(aclk),
        .rst_n(aresetn),
        .start(writer_start),
        .conv_mode(conv2d_mode),
        .base_addr(c_base + (m0 * c_stride) + ((n0 + write_array_idx * 16) * 32'd4)),
        .row_stride(c_stride),
        .output_base_addr(c_base), .out_h(conv_out_h), .out_w(conv_out_w), .m0(m0), .n0(n0 + write_array_idx * 16),
        .tile_rows(m_len),
        .tile_cols(writer_n_len),
        .busy(writer_busy),
        .done(writer_done),
        .error(writer_error),
        .tile_read_req(writer_tile_read_req),
        .tile_read_addr(writer_tile_read_addr),
        .tile_read_valid(writer_tile_read_valid),
        .tile_read_data(writer_tile_read_data),
        .write_req_valid(writer_write_req_valid),
        .write_req_last(writer_write_req_last),
        .write_req_ready(writer_write_req_ready),
        .write_req_addr(writer_write_req_addr),
        .write_req_data(writer_write_req_data)
    );

    requantize_ddr requantizer (
        .clk(aclk),
        .rst_n(aresetn),
        .start(rq_start),
        .input_base_addr(a_base),
        .output_base_addr(c_base),
        .element_count({16'd0, dim_m}),
        .multiplier(rq_multiplier),
        .right_shift(rq_right_shift),
        .output_zero_point(rq_output_zero_point),
        .busy(rq_busy),
        .done(rq_done),
        .error(rq_error),
        .read_req_valid(rq_read_req_valid),
        .read_req_ready(rq_read_req_ready),
        .read_req_addr(rq_read_req_addr),
        .read_data_valid(rq_read_data_valid),
        .read_data_ready(rq_read_data_ready),
        .read_data(rq_read_data),
        .write_req_valid(rq_write_req_valid),
        .write_req_ready(rq_write_req_ready),
        .write_req_addr(rq_write_req_addr),
        .write_req_data(rq_write_req_data),
        .write_req_strb(rq_write_req_strb)
    );

    activation_ddr_parallel activation (
        .clk(aclk), .rst_n(aresetn), .start(activation_start), .leaky_mode(activation_leaky_mode), .leaky_shift(rq_right_shift[2:0]),
        .input_base_addr(a_base), .output_base_addr(c_base), .element_count({16'd0, dim_m}),
        .busy(activation_busy), .done(activation_done), .error(activation_error),
        .read_req_valid(activation_read_req_valid), .read_req_ready(activation_read_req_ready), .read_req_addr(activation_read_req_addr),
        .read_data_valid(activation_read_data_valid), .read_data_ready(activation_read_data_ready), .read_data(activation_read_data),
        .write_req_valid(activation_write_req_valid), .write_req_ready(activation_write_req_ready), .write_req_addr(activation_write_req_addr), .write_req_data(activation_write_req_data), .write_req_strb(activation_write_req_strb)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            op_state <= OP_IDLE;
            ctrl_start_d <= 1'b0;
            core_start <= 1'b0;
            done_r <= 1'b0;
            error_r <= 1'b0;
            write_complete_pending <= 1'b0;
            zero_idx <= '0;
            accum_idx <= '0;
            zero_b_array_idx <= '0;
            load_b_array_idx <= '0;
            accum_array_idx <= '0;
            write_array_idx <= '0;
        end else begin
            ctrl_start_d <= ctrl_word[0];
            core_start <= 1'b0;
            if (op_state != OP_WRITE_C) begin
                write_complete_pending <= 1'b0;
            end else if (writer_done && write_array_idx == active_arrays - 1'b1) begin
                write_complete_pending <= 1'b1;
            end

            // Configuration validation belongs to gemm_controller at start.
            // Register reset values are intentionally not a fault while idle.
            if (controller_error || loader_error || writer_error || rq_error || activation_error || bias_preload_error || memio_error) begin
                error_r <= 1'b1;
            end

            if (controller_done) begin
                done_r <= 1'b1;
            end
            if (rq_done) begin
                done_r <= 1'b1;
            end
            if (activation_done) begin
                done_r <= 1'b1;
            end
            if (!ctrl_word[0]) begin
                done_r <= 1'b0;
            end

            unique case (op_state)
                OP_IDLE: begin
                    zero_idx <= '0;
                    accum_idx <= '0;
                    zero_b_array_idx <= '0;
                    load_b_array_idx <= '0;
                    accum_array_idx <= '0;
                    write_array_idx <= '0;
                    if (start_pulse) begin
                        error_r <= 1'b0;
                    end else if (load_a_start) begin
                        op_state <= OP_ZERO_A;
                    end else if (load_b_start) begin
                        op_state <= OP_ZERO_B;
                    end else if (compute_start) begin
                        if (conv2d_mode && clear_accumulators && conv_bias_base != 0) op_state <= OP_BIAS_PRELOAD;
                        else begin core_start <= 1'b1; op_state <= OP_COMPUTE_WAIT; end
                    end else if (write_c_start) begin
                        op_state <= OP_WRITE_C;
                    end
                end

                OP_ZERO_A: begin
                    zero_idx <= zero_idx + 8'd1;
                    if (zero_idx == 8'hff) begin
                        op_state <= OP_LOAD_A;
                    end
                end

                OP_LOAD_A: begin
                    if (loader_done) begin
                        op_state <= OP_IDLE;
                    end
                end

                OP_ZERO_B: begin
                    zero_idx <= zero_idx + 8'd1;
                    if (zero_idx == 8'hff) begin
                        zero_idx <= '0;
                        if (zero_b_array_idx + 1'b1 < active_arrays) begin
                            zero_b_array_idx <= zero_b_array_idx + 1'b1;
                        end else begin
                            load_b_array_idx <= '0;
                            op_state <= OP_LOAD_B_NEXT;
                        end
                    end
                end

                OP_LOAD_B: begin
                    if (loader_done) begin
                        if (load_b_array_idx + 1'b1 < active_arrays) begin
                            load_b_array_idx <= load_b_array_idx + 1'b1;
                            op_state <= OP_LOAD_B_NEXT;
                        end else begin
                            op_state <= OP_IDLE;
                        end
                    end
                end

                OP_LOAD_B_NEXT: op_state <= OP_LOAD_B;

                OP_BIAS_PRELOAD: begin
                    if (bias_preload_done) begin
                        core_start <= 1'b1;
                        op_state <= OP_COMPUTE_WAIT;
                    end
                end

                OP_COMPUTE_WAIT: begin
                    if (core_done) begin
                        accum_idx <= '0;
                        accum_array_idx <= '0;
                        op_state <= OP_ACCUMULATE;
                    end
                end

                OP_ACCUMULATE: begin
                    if (accum_op_ready) begin
                        if (accum_idx == 8'hff) begin
                            accum_idx <= '0;
                            if (accum_array_idx + 1'b1 < active_arrays) begin
                                accum_array_idx <= accum_array_idx + 1'b1;
                            end else begin
                                op_state <= OP_IDLE;
                            end
                        end else begin
                            accum_idx <= accum_idx + 8'd1;
                        end
                    end
                end

                OP_WRITE_C: begin
                    if (write_array_idx + 1'b1 < active_arrays && writer_done) begin
                        write_array_idx <= write_array_idx + 1'b1;
                    end else if (write_array_idx + 1'b1 >= active_arrays && write_c_done) begin
                        op_state <= OP_IDLE;
                    end
                end

                default: op_state <= OP_IDLE;
            endcase
        end
    end

endmodule
