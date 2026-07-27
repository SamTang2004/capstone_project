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

    typedef enum logic [3:0] {
        LOADER_LAUNCH_IDLE,
        LOADER_LAUNCH_MUL0,
        LOADER_LAUNCH_MUL1,
        LOADER_LAUNCH_ADD01,
        LOADER_LAUNCH_MUL2,
        LOADER_LAUNCH_MUL3,
        LOADER_LAUNCH_ADD23,
        LOADER_LAUNCH_COMBINE,
        LOADER_LAUNCH_PRODUCT,
        LOADER_LAUNCH_BASE,
        LOADER_LAUNCH_FINAL,
        LOADER_LAUNCH_START,
        LOADER_LAUNCH_WAIT
    } loader_launch_state_t;

    typedef enum logic [3:0] {
        WRITER_LAUNCH_IDLE,
        WRITER_LAUNCH_ROW_CHUNKS,
        WRITER_LAUNCH_ROW_PAIRS,
        WRITER_LAUNCH_ROW_COMBINE,
        WRITER_LAUNCH_GEOMETRY,
        WRITER_LAUNCH_PRODUCT,
        WRITER_LAUNCH_BASE,
        WRITER_LAUNCH_FINAL,
        WRITER_LAUNCH_START,
        WRITER_LAUNCH_WAIT
    } writer_launch_state_t;

    op_state_t op_state;

    logic [31:0] ctrl_word;
    logic [31:0] status_word;
    logic [32*16-1:0] regs_flat;
    logic ctrl_start_d;
    logic start_pulse;
    logic command_stage1_valid;
    logic command_stage2_valid;
    logic dispatch_pulse;
    logic done_r;
    logic error_r;
    logic write_complete_pending;

    wire [31:0] mmio_a_base = regs_flat[2*32 +: 32];
    wire [31:0] mmio_b_base = regs_flat[3*32 +: 32];
    wire [31:0] mmio_c_base = regs_flat[4*32 +: 32];
    wire [31:0] mmio_a_stride = regs_flat[5*32 +: 32];
    wire [31:0] mmio_b_stride = regs_flat[6*32 +: 32];
    wire [31:0] mmio_c_stride = regs_flat[7*32 +: 32];
    wire [15:0] mmio_dim_m = regs_flat[8*32 +: 16];
    wire [15:0] mmio_dim_n = regs_flat[9*32 +: 16];
    wire [15:0] mmio_dim_k = regs_flat[10*32 +: 16];
    wire [4:0] mmio_tile_size = regs_flat[11*32 +: 5];
    wire [7:0] mmio_array_count = regs_flat[12*32 +: 8];
    wire signed [31:0] mmio_rq_multiplier = regs_flat[13*32 +: 32];
    wire [31:0] mmio_rq_right_shift = regs_flat[14*32 +: 32];
    wire signed [31:0] mmio_rq_output_zero_point = regs_flat[15*32 +: 32];

    // Snapshot the software-visible register file before dispatch. These
    // registers isolate the high-fanout MMIO bank from every datapath.
    logic [31:0] a_base, b_base, c_base;
    logic [31:0] a_stride, b_stride, c_stride;
    logic [15:0] dim_m, dim_n, dim_k;
    logic [4:0] tile_size;
    logic [7:0] array_count;
    logic signed [31:0] rq_multiplier;
    logic [31:0] rq_right_shift;
    logic signed [31:0] rq_output_zero_point;
    logic requantize_mode, conv2d_mode, activation_mode, activation_leaky_mode;
    logic maxpool_mode, upsample_mode, concat_mode, avgpool_mode;
    wire tensor_mode = maxpool_mode || upsample_mode || concat_mode;
    wire [31:0] conv_bias_base = rq_output_zero_point;
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
    logic [31:0] conv_spatial_r;
    logic [12:0] conv_kernel_area_r;
    logic [15:0] controller_dim_m, controller_dim_n, controller_dim_k;
    logic [4:0] controller_tile_size;
    logic [7:0] controller_array_count;
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
    logic loader_start_request;
    logic loader_matrix_select;
    logic [31:0] loader_base_addr;
    logic [31:0] loader_row_stride;
    logic [4:0] loader_rows;
    logic [4:0] loader_cols;
    loader_launch_state_t loader_launch_state;
    logic [31:0] loader_base_seed_r, loader_row_offset_r, loader_column_offset_r, loader_base_sum_r;
    logic [15:0] loader_offset_index_r;
    logic [31:0] loader_offset_stride_r;
    logic [23:0] loader_offset_chunk0_r, loader_offset_chunk1_r;
    logic [31:0] loader_offset_pair_low_r, loader_offset_pair_high_r;
    logic loader_busy;
    logic loader_done;
    logic loader_error;
    logic loader_read_req_valid;
    logic loader_read_req_ready;
    logic [31:0] loader_read_req_addr;
    logic [4:0] loader_read_req_len;
    logic loader_read_data_valid;
    logic loader_read_data_ready;
    logic loader_read_data_last;
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
    logic writer_start_request;
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
    writer_launch_state_t writer_launch_state;
    logic [31:0] writer_base_addr_r, writer_base_seed_r;
    logic [31:0] writer_row_offset_r, writer_column_offset_r, writer_base_sum_r;
    logic [15:0] writer_m0_seed_r;
    logic [31:0] writer_stride_seed_r;
    logic writer_conv2d_seed_r;
    (* use_dsp = "no" *) logic [23:0] writer_row_chunk0_r, writer_row_chunk1_r;
    (* use_dsp = "no" *) logic [23:0] writer_row_chunk2_r, writer_row_chunk3_r;
    logic [31:0] writer_row_pair_low_r, writer_row_pair_high_r;
    logic [15:0] writer_n0_r;
    logic [4:0] writer_rows_r, writer_cols_r;
    logic [15:0] writer_n0_seed_r, writer_col_offset_seed_r, writer_n_group_cols_seed_r;

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

    logic tensor_start, tensor_busy, tensor_done, tensor_error;
    logic tensor_read_req_valid, tensor_read_req_ready, tensor_read_data_valid, tensor_read_data_ready;
    logic [31:0] tensor_read_req_addr;
    logic [7:0] tensor_read_data;
    logic tensor_write_req_valid, tensor_write_req_ready;
    logic [31:0] tensor_write_req_addr, tensor_write_req_data;
    logic [3:0] tensor_write_req_strb;

    logic avgpool_start, avgpool_busy, avgpool_done, avgpool_error;
    logic avgpool_read_req_valid, avgpool_read_req_ready, avgpool_read_data_valid, avgpool_read_data_ready;
    logic [31:0] avgpool_read_req_addr;
    logic [7:0] avgpool_read_data;
    logic avgpool_write_req_valid, avgpool_write_req_ready;
    logic [31:0] avgpool_write_req_addr, avgpool_write_req_data;
    logic [3:0] avgpool_write_req_strb;

    logic memio_read_req_valid;
    logic memio_read_req_ready;
    logic [31:0] memio_read_req_addr;
    logic [4:0] memio_read_req_len;
    logic memio_read_data_valid;
    logic memio_read_data_ready;
    logic memio_read_data_last;
    logic [7:0] memio_read_data;
    logic memio_busy;
    logic memio_error;
    logic memio_write_req_ready;

    // DDR read data can return after the issuing operator changes state.
    // Keep its destination stable from request acceptance through response
    // consumption instead of deriving it from the current busy signals.
    typedef enum logic [2:0] {
        READ_OWNER_LOADER,
        READ_OWNER_REQUANT,
        READ_OWNER_ACTIVATION,
        READ_OWNER_BIAS,
        READ_OWNER_TENSOR,
        READ_OWNER_AVGPOOL
    } read_owner_t;
    read_owner_t read_owner;
    logic read_owner_valid;

    wire [4:0] b_len = (n_group_cols > (load_b_array_idx * 16 + 16)) ? 5'd16 : (n_group_cols - load_b_array_idx * 16);
    wire [2:0] writer_request_array = writer_done ? write_array_idx + 1'b1 : write_array_idx;
    wire [15:0] writer_request_col_offset = {13'd0, writer_request_array} << 4;

    assign start_pulse = ctrl_word[0] && !ctrl_start_d;
    assign controller_start = dispatch_pulse && !requantize_mode && !activation_mode && !tensor_mode && !avgpool_mode;
    assign rq_start = dispatch_pulse && requantize_mode;
    assign activation_start = dispatch_pulse && activation_mode;
    assign tensor_start = dispatch_pulse && tensor_mode;
    assign avgpool_start = dispatch_pulse && avgpool_mode;

    // MMIO decode pipeline:
    //   1. atomically snapshot the register file and operation mode;
    //   2. calculate independent Conv2D products;
    //   3. register the controller-ready dimensions;
    //   4. pulse the selected operator after every input is stable.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            command_stage1_valid <= 1'b0;
            command_stage2_valid <= 1'b0;
            dispatch_pulse <= 1'b0;
            a_base <= '0;
            b_base <= '0;
            c_base <= '0;
            a_stride <= '0;
            b_stride <= '0;
            c_stride <= '0;
            dim_m <= '0;
            dim_n <= '0;
            dim_k <= '0;
            tile_size <= '0;
            array_count <= '0;
            rq_multiplier <= '0;
            rq_right_shift <= '0;
            rq_output_zero_point <= '0;
            requantize_mode <= 1'b0;
            conv2d_mode <= 1'b0;
            activation_mode <= 1'b0;
            activation_leaky_mode <= 1'b0;
            maxpool_mode <= 1'b0;
            upsample_mode <= 1'b0;
            concat_mode <= 1'b0;
            avgpool_mode <= 1'b0;
            conv_spatial_r <= '0;
            conv_kernel_area_r <= '0;
            controller_dim_m <= '0;
            controller_dim_n <= '0;
            controller_dim_k <= '0;
            controller_tile_size <= '0;
            controller_array_count <= '0;
        end else begin
            command_stage1_valid <= start_pulse;
            command_stage2_valid <= command_stage1_valid;
            dispatch_pulse <= command_stage2_valid;

            if (start_pulse) begin
                a_base <= mmio_a_base;
                b_base <= mmio_b_base;
                c_base <= mmio_c_base;
                a_stride <= mmio_a_stride;
                b_stride <= mmio_b_stride;
                c_stride <= mmio_c_stride;
                dim_m <= mmio_dim_m;
                dim_n <= mmio_dim_n;
                dim_k <= mmio_dim_k;
                tile_size <= mmio_tile_size;
                array_count <= mmio_array_count;
                rq_multiplier <= mmio_rq_multiplier;
                rq_right_shift <= mmio_rq_right_shift;
                rq_output_zero_point <= mmio_rq_output_zero_point;
                requantize_mode <= ctrl_word[1];
                conv2d_mode <= ctrl_word[2];
                activation_mode <= ctrl_word[3];
                activation_leaky_mode <= ctrl_word[4];
                maxpool_mode <= ctrl_word[5];
                upsample_mode <= ctrl_word[6];
                concat_mode <= ctrl_word[7];
                avgpool_mode <= ctrl_word[9];
            end

            if (command_stage1_valid) begin
                conv_spatial_r <= dim_n * dim_k;
                conv_kernel_area_r <= tile_size * array_count;
            end

            if (command_stage2_valid) begin
                controller_dim_m <= conv2d_mode ? conv_spatial_r[15:0] : dim_m;
                controller_dim_n <= conv2d_mode ? dim_m : dim_n;
                controller_dim_k <= conv2d_mode ? conv_input_c * conv_kernel_area_r : dim_k;
                controller_tile_size <= conv2d_mode ? 5'd16 : tile_size;
                controller_array_count <= conv2d_mode ? 8'd1 : array_count;
            end
        end
    end
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
        // DONE means the operator has generated its final request. Keep BUSY
        // asserted until the shared HP0 path has drained its final response.
        start_pulse || command_stage1_valid || command_stage2_valid || dispatch_pulse ||
        controller_busy || rq_busy || activation_busy || tensor_busy || avgpool_busy ||
        op_state != OP_IDLE || memio_busy
    };

    assign load_a_done = (op_state == OP_LOAD_A) && loader_done;
    assign load_b_done = (op_state == OP_LOAD_B) && loader_done;
    assign compute_done = (op_state == OP_ACCUMULATE) && accum_op_valid && accum_op_ready && accum_idx == 8'hff && accum_array_idx == active_arrays - 1'b1;
    assign write_c_done = (op_state == OP_WRITE_C) && write_complete_pending && !memio_busy;

    assign loader_start_request = ((op_state == OP_ZERO_A) && zero_idx == 8'hff) ||
                                  (op_state == OP_LOAD_B_NEXT);

    // Build the normal GEMM tile base over registered stages before the
    // TileLoader captures it.  This removes the m0/k0 stride arithmetic and
    // control-state fanout from the loader's configuration-register setup path.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            loader_launch_state <= LOADER_LAUNCH_IDLE;
            loader_start <= 1'b0;
            loader_matrix_select <= 1'b0;
            loader_base_addr <= '0;
            loader_row_stride <= '0;
            loader_rows <= '0;
            loader_cols <= '0;
            loader_base_seed_r <= '0;
            loader_row_offset_r <= '0;
            loader_column_offset_r <= '0;
            loader_base_sum_r <= '0;
            loader_offset_index_r <= '0;
            loader_offset_stride_r <= '0;
            loader_offset_chunk0_r <= '0;
            loader_offset_chunk1_r <= '0;
            loader_offset_pair_low_r <= '0;
            loader_offset_pair_high_r <= '0;
        end else begin
            loader_start <= 1'b0;
            case (loader_launch_state)
                LOADER_LAUNCH_IDLE: begin
                    if (loader_start_request) begin
                        loader_matrix_select <= (op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT);
                        loader_base_seed_r <= ((op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT)) ? b_base : a_base;
                        loader_row_stride <= ((op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT)) ? b_stride : a_stride;
                        loader_rows <= ((op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT)) ? k_len : m_len;
                        loader_cols <= ((op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT)) ? b_len : k_len;
                        if ((op_state == OP_ZERO_B) || (op_state == OP_LOAD_B) || (op_state == OP_LOAD_B_NEXT)) begin
                            loader_offset_index_r <= k0;
                            loader_offset_stride_r <= b_stride;
                            loader_column_offset_r <= n0 + (load_b_array_idx << 4);
                        end else begin
                            loader_offset_index_r <= m0;
                            loader_offset_stride_r <= a_stride;
                            loader_column_offset_r <= k0;
                        end
                        loader_launch_state <= LOADER_LAUNCH_MUL0;
                    end
                end
                LOADER_LAUNCH_MUL0: begin
                    loader_offset_chunk0_r <= loader_offset_stride_r[7:0] * loader_offset_index_r;
                    loader_launch_state <= LOADER_LAUNCH_MUL1;
                end
                LOADER_LAUNCH_MUL1: begin
                    loader_offset_chunk1_r <= loader_offset_stride_r[15:8] * loader_offset_index_r;
                    loader_launch_state <= LOADER_LAUNCH_ADD01;
                end
                LOADER_LAUNCH_ADD01: begin
                    loader_offset_pair_low_r <= {8'd0, loader_offset_chunk0_r} + ({8'd0, loader_offset_chunk1_r} << 8);
                    loader_launch_state <= LOADER_LAUNCH_MUL2;
                end
                LOADER_LAUNCH_MUL2: begin
                    loader_offset_chunk0_r <= loader_offset_stride_r[23:16] * loader_offset_index_r;
                    loader_launch_state <= LOADER_LAUNCH_MUL3;
                end
                LOADER_LAUNCH_MUL3: begin
                    loader_offset_chunk1_r <= loader_offset_stride_r[31:24] * loader_offset_index_r;
                    loader_launch_state <= LOADER_LAUNCH_ADD23;
                end
                LOADER_LAUNCH_ADD23: begin
                    loader_offset_pair_high_r <= {8'd0, loader_offset_chunk0_r} + ({8'd0, loader_offset_chunk1_r} << 8);
                    loader_launch_state <= LOADER_LAUNCH_COMBINE;
                end
                LOADER_LAUNCH_COMBINE: begin
                    loader_row_offset_r <= loader_offset_pair_low_r + (loader_offset_pair_high_r << 16);
                    loader_launch_state <= LOADER_LAUNCH_PRODUCT;
                end
                LOADER_LAUNCH_PRODUCT: begin
                    loader_base_sum_r <= loader_base_seed_r + loader_row_offset_r;
                    loader_launch_state <= LOADER_LAUNCH_BASE;
                end
                LOADER_LAUNCH_BASE: begin
                    loader_base_addr <= loader_base_sum_r + loader_column_offset_r;
                    loader_launch_state <= LOADER_LAUNCH_FINAL;
                end
                LOADER_LAUNCH_FINAL: loader_launch_state <= LOADER_LAUNCH_START;
                LOADER_LAUNCH_START: begin
                    loader_start <= 1'b1;
                    loader_launch_state <= LOADER_LAUNCH_WAIT;
                end
                LOADER_LAUNCH_WAIT: begin
                    if (!loader_start_request) loader_launch_state <= LOADER_LAUNCH_IDLE;
                end
                default: loader_launch_state <= LOADER_LAUNCH_IDLE;
            endcase
        end
    end

    assign memio_read_req_valid = avgpool_busy ? avgpool_read_req_valid : (tensor_busy ? tensor_read_req_valid : (bias_preload_busy ? bias_read_req_valid : (activation_busy ? activation_read_req_valid : (rq_busy ? rq_read_req_valid : loader_read_req_valid))));
    assign loader_read_req_ready = !avgpool_busy && !tensor_busy && !bias_preload_busy && !activation_busy && !rq_busy && memio_read_req_ready;
    assign rq_read_req_ready = !avgpool_busy && !tensor_busy && !bias_preload_busy && !activation_busy && rq_busy && memio_read_req_ready;
    assign activation_read_req_ready = !avgpool_busy && !tensor_busy && !bias_preload_busy && activation_busy && memio_read_req_ready;
    assign bias_read_req_ready = !avgpool_busy && !tensor_busy && bias_preload_busy && memio_read_req_ready;
    assign tensor_read_req_ready = !avgpool_busy && tensor_busy && memio_read_req_ready;
    assign avgpool_read_req_ready = avgpool_busy && memio_read_req_ready;
    assign memio_read_req_addr = avgpool_busy ? avgpool_read_req_addr : (tensor_busy ? tensor_read_req_addr : (bias_preload_busy ? bias_read_req_addr : (activation_busy ? activation_read_req_addr : (rq_busy ? rq_read_req_addr : loader_read_req_addr))));
    assign memio_read_req_len = (!avgpool_busy && !tensor_busy && !bias_preload_busy && !activation_busy && !rq_busy) ? loader_read_req_len : 5'd1;
    assign loader_read_data_valid = read_owner_valid && read_owner == READ_OWNER_LOADER && memio_read_data_valid;
    assign rq_read_data_valid = read_owner_valid && read_owner == READ_OWNER_REQUANT && memio_read_data_valid;
    assign activation_read_data_valid = read_owner_valid && read_owner == READ_OWNER_ACTIVATION && memio_read_data_valid;
    assign bias_read_data_valid = read_owner_valid && read_owner == READ_OWNER_BIAS && memio_read_data_valid;
    assign tensor_read_data_valid = read_owner_valid && read_owner == READ_OWNER_TENSOR && memio_read_data_valid;
    assign avgpool_read_data_valid = read_owner_valid && read_owner == READ_OWNER_AVGPOOL && memio_read_data_valid;
    always_comb begin
        unique case (read_owner)
            READ_OWNER_REQUANT:    memio_read_data_ready = rq_busy ? rq_read_data_ready : 1'b1;
            READ_OWNER_ACTIVATION: memio_read_data_ready = activation_busy ? activation_read_data_ready : 1'b1;
            READ_OWNER_BIAS:       memio_read_data_ready = bias_preload_busy ? bias_read_data_ready : 1'b1;
            READ_OWNER_TENSOR:     memio_read_data_ready = tensor_busy ? tensor_read_data_ready : 1'b1;
            READ_OWNER_AVGPOOL:    memio_read_data_ready = avgpool_busy ? avgpool_read_data_ready : 1'b1;
            default:               memio_read_data_ready = loader_busy ? loader_read_data_ready : 1'b1;
        endcase
        if (!read_owner_valid) memio_read_data_ready = 1'b0;
    end
    assign loader_read_data = memio_read_data;
    assign loader_read_data_last = memio_read_data_last;
    assign rq_read_data = memio_read_data;
    assign activation_read_data = memio_read_data;
    assign tensor_read_data = memio_read_data;
    assign avgpool_read_data = memio_read_data;
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

    assign writer_start_request = write_c_start ||
                                  (op_state == OP_WRITE_C && writer_done && write_array_idx < active_arrays - 1'b1);
    assign writer_write_req_strb = 4'hf;

    // Register the controller-derived C tile base before TileWriter launch.
    // This removes m0/n0 stride arithmetic from the writer's IDLE capture
    // cone and also selects the next array index before a chained launch.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            writer_launch_state <= WRITER_LAUNCH_IDLE;
            writer_start <= 1'b0;
            writer_base_addr_r <= '0;
            writer_base_seed_r <= '0;
            writer_row_offset_r <= '0;
            writer_column_offset_r <= '0;
            writer_base_sum_r <= '0;
            writer_m0_seed_r <= '0;
            writer_stride_seed_r <= '0;
            writer_conv2d_seed_r <= 1'b0;
            writer_row_chunk0_r <= '0;
            writer_row_chunk1_r <= '0;
            writer_row_chunk2_r <= '0;
            writer_row_chunk3_r <= '0;
            writer_row_pair_low_r <= '0;
            writer_row_pair_high_r <= '0;
            writer_n0_r <= '0;
            writer_rows_r <= '0;
            writer_cols_r <= '0;
            writer_n0_seed_r <= '0;
            writer_col_offset_seed_r <= '0;
            writer_n_group_cols_seed_r <= '0;
        end else begin
            writer_start <= 1'b0;
            case (writer_launch_state)
                WRITER_LAUNCH_IDLE: begin
                    if (writer_start_request) begin
                        writer_base_seed_r <= c_base;
                        writer_n0_seed_r <= n0;
                        writer_col_offset_seed_r <= writer_request_col_offset;
                        writer_n_group_cols_seed_r <= n_group_cols;
                        writer_rows_r <= m_len;
                        writer_m0_seed_r <= m0;
                        writer_stride_seed_r <= c_stride;
                        writer_conv2d_seed_r <= conv2d_mode;
                        if (conv2d_mode) begin
                            writer_column_offset_r <= '0;
                        end else begin
                            writer_column_offset_r <= (n0 + writer_request_col_offset) << 2;
                        end
                        writer_launch_state <= WRITER_LAUNCH_ROW_CHUNKS;
                    end
                end
                WRITER_LAUNCH_ROW_CHUNKS: begin
                    if (writer_conv2d_seed_r) begin
                        writer_row_chunk0_r <= '0;
                        writer_row_chunk1_r <= '0;
                        writer_row_chunk2_r <= '0;
                        writer_row_chunk3_r <= '0;
                    end else begin
                        writer_row_chunk0_r <= writer_m0_seed_r * writer_stride_seed_r[7:0];
                        writer_row_chunk1_r <= writer_m0_seed_r * writer_stride_seed_r[15:8];
                        writer_row_chunk2_r <= writer_m0_seed_r * writer_stride_seed_r[23:16];
                        writer_row_chunk3_r <= writer_m0_seed_r * writer_stride_seed_r[31:24];
                    end
                    writer_launch_state <= WRITER_LAUNCH_ROW_PAIRS;
                end
                WRITER_LAUNCH_ROW_PAIRS: begin
                    writer_row_pair_low_r <= writer_row_chunk0_r + (writer_row_chunk1_r << 8);
                    writer_row_pair_high_r <= writer_row_chunk2_r + (writer_row_chunk3_r << 8);
                    writer_launch_state <= WRITER_LAUNCH_ROW_COMBINE;
                end
                WRITER_LAUNCH_ROW_COMBINE: begin
                    writer_row_offset_r <= writer_row_pair_low_r + (writer_row_pair_high_r << 16);
                    writer_launch_state <= WRITER_LAUNCH_GEOMETRY;
                end
                WRITER_LAUNCH_GEOMETRY: begin
                    writer_n0_r <= writer_n0_seed_r + writer_col_offset_seed_r;
                    writer_cols_r <= (writer_n_group_cols_seed_r > writer_col_offset_seed_r + 16) ?
                                     5'd16 : writer_n_group_cols_seed_r - writer_col_offset_seed_r;
                    writer_launch_state <= WRITER_LAUNCH_PRODUCT;
                end
                WRITER_LAUNCH_PRODUCT: begin
                    writer_base_sum_r <= writer_base_seed_r + writer_row_offset_r;
                    writer_launch_state <= WRITER_LAUNCH_BASE;
                end
                WRITER_LAUNCH_BASE: begin
                    writer_base_addr_r <= writer_base_sum_r + writer_column_offset_r;
                    writer_launch_state <= WRITER_LAUNCH_FINAL;
                end
                WRITER_LAUNCH_FINAL: writer_launch_state <= WRITER_LAUNCH_START;
                WRITER_LAUNCH_START: begin
                    writer_start <= 1'b1;
                    writer_launch_state <= WRITER_LAUNCH_WAIT;
                end
                WRITER_LAUNCH_WAIT: begin
                    if (!writer_start_request) writer_launch_state <= WRITER_LAUNCH_IDLE;
                end
                default: writer_launch_state <= WRITER_LAUNCH_IDLE;
            endcase
        end
    end
    axi_lite_control #(.ADDR_WIDTH(6)) control (
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
        .diagnostics_flat('0),
        .ctrl_word(ctrl_word),
        .regs_flat(regs_flat)
    );

    gemm_controller #(
        .MAX_ARRAYS(MAX_ARRAYS)
    ) controller (
        .clk(aclk),
        .rst_n(aresetn),
        .start(controller_start),
        .M(controller_dim_m),
        .N(controller_dim_n),
        .K(controller_dim_k),
        .tile_size(controller_tile_size),
        .array_count(controller_array_count),
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
        .invalidate_cache(1'b0),
        .read_req_valid(memio_read_req_valid),
        .read_req_ready(memio_read_req_ready),
        .read_req_addr(memio_read_req_addr),
        .read_req_len(memio_read_req_len),
        .read_data_valid(memio_read_data_valid),
        .read_data_ready(memio_read_data_ready),
        .read_data(memio_read_data),
        .read_data_last(memio_read_data_last),
        .write_req_valid(avgpool_busy ? avgpool_write_req_valid : (tensor_busy ? tensor_write_req_valid : (activation_busy ? activation_write_req_valid : (rq_busy ? rq_write_req_valid : writer_write_req_valid)))),
        .write_req_ready(memio_write_req_ready),
        .write_req_addr(avgpool_busy ? avgpool_write_req_addr : (tensor_busy ? tensor_write_req_addr : (activation_busy ? activation_write_req_addr : (rq_busy ? rq_write_req_addr : writer_write_req_addr)))),
        .write_req_data(avgpool_busy ? avgpool_write_req_data : (tensor_busy ? tensor_write_req_data : (activation_busy ? activation_write_req_data : (rq_busy ? rq_write_req_data : writer_write_req_data)))),
        .write_req_strb(avgpool_busy ? avgpool_write_req_strb : (tensor_busy ? tensor_write_req_strb : (activation_busy ? activation_write_req_strb : (rq_busy ? rq_write_req_strb : writer_write_req_strb)))),
        .write_req_last(avgpool_busy ? 1'b1 : (tensor_busy ? 1'b1 : (activation_busy ? 1'b1 : (rq_busy ? 1'b1 : writer_write_req_last)))),
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

    assign writer_write_req_ready = !avgpool_busy && !tensor_busy && !activation_busy && !rq_busy && memio_write_req_ready;
    assign rq_write_req_ready = !avgpool_busy && !tensor_busy && !activation_busy && rq_busy && memio_write_req_ready;
    assign activation_write_req_ready = !avgpool_busy && !tensor_busy && activation_busy && memio_write_req_ready;
    assign tensor_write_req_ready = !avgpool_busy && tensor_busy && memio_write_req_ready;
    assign avgpool_write_req_ready = avgpool_busy && memio_write_req_ready;

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
        .read_req_len(loader_read_req_len),
        .read_data_valid(loader_read_data_valid),
        .read_data_ready(loader_read_data_ready),
        .read_data(loader_read_data),
        .read_data_last(loader_read_data_last),
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
        .base_addr(writer_base_addr_r),
        .row_stride(c_stride),
        .output_base_addr(c_base), .out_h(conv_out_h), .out_w(conv_out_w), .m0(m0), .n0(writer_n0_r),
        .tile_rows(writer_rows_r),
        .tile_cols(writer_cols_r),
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

    // Tensor-op register overlay: A_STRIDE=N, B_STRIDE=C0, C_STRIDE=H,
    // M=W, N=out_H, K=out_W.  For concat, RQ_MULTIPLIER/RQ_RIGHT_SHIFT
    // hold C1/C2; for pool/upsample they hold stride/scale and padding.
    cnn_tensor_ops_ddr tensor_ops (
        .clk(aclk), .rst_n(aresetn), .start(tensor_start),
        .mode(maxpool_mode ? 2'd1 : (upsample_mode ? 2'd2 : 2'd3)),
        .input0_base_addr(a_base), .input1_base_addr(b_base), .input2_base_addr(rq_output_zero_point), .output_base_addr(c_base),
        .n(a_stride[15:0]), .channels0(b_stride[15:0]),
        .channels1(concat_mode ? rq_multiplier[15:0] : 16'd0), .channels2(concat_mode ? rq_right_shift[15:0] : 16'd0),
        .input_h(c_stride[15:0]), .input_w(dim_m), .output_h(dim_n), .output_w(dim_k),
        .kernel_h({3'd0, tile_size}), .kernel_w(array_count), .stride_h(rq_multiplier[7:0]), .stride_w(rq_multiplier[15:8]),
        .pad_h(rq_right_shift[7:0]), .pad_w(rq_right_shift[15:8]),
        .busy(tensor_busy), .done(tensor_done), .error(tensor_error),
        .read_req_valid(tensor_read_req_valid), .read_req_ready(tensor_read_req_ready), .read_req_addr(tensor_read_req_addr),
        .read_data_valid(tensor_read_data_valid), .read_data_ready(tensor_read_data_ready), .read_data(tensor_read_data),
        .write_req_valid(tensor_write_req_valid), .write_req_ready(tensor_write_req_ready), .write_req_addr(tensor_write_req_addr), .write_req_data(tensor_write_req_data), .write_req_strb(tensor_write_req_strb)
    );

    // GlobalAveragePool uses this same engine with kernel_h=H, kernel_w=W,
    // stride=1, and a 1x1 output shape.
    avgpool_ddr avgpool_ops (
        .clk(aclk), .rst_n(aresetn), .start(avgpool_start),
        .input_base_addr(a_base), .output_base_addr(c_base),
        .n(a_stride[15:0]), .channels(b_stride[15:0]), .input_h(c_stride[15:0]), .input_w(dim_m),
        .output_h(dim_n), .output_w(dim_k), .kernel_h({3'd0, tile_size}), .kernel_w(array_count),
        .stride_h(rq_multiplier[7:0]), .stride_w(rq_multiplier[15:8]), .pad_h(rq_right_shift[7:0]), .pad_w(rq_right_shift[15:8]),
        .busy(avgpool_busy), .done(avgpool_done), .error(avgpool_error),
        .read_req_valid(avgpool_read_req_valid), .read_req_ready(avgpool_read_req_ready), .read_req_addr(avgpool_read_req_addr),
        .read_data_valid(avgpool_read_data_valid), .read_data_ready(avgpool_read_data_ready), .read_data(avgpool_read_data),
        .write_req_valid(avgpool_write_req_valid), .write_req_ready(avgpool_write_req_ready), .write_req_addr(avgpool_write_req_addr), .write_req_data(avgpool_write_req_data), .write_req_strb(avgpool_write_req_strb)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            op_state <= OP_IDLE;
            ctrl_start_d <= 1'b0;
            core_start <= 1'b0;
            done_r <= 1'b0;
            error_r <= 1'b0;
            write_complete_pending <= 1'b0;
            read_owner <= READ_OWNER_LOADER;
            read_owner_valid <= 1'b0;
            zero_idx <= '0;
            accum_idx <= '0;
            zero_b_array_idx <= '0;
            load_b_array_idx <= '0;
            accum_array_idx <= '0;
            write_array_idx <= '0;
        end else begin
            ctrl_start_d <= ctrl_word[0];
            core_start <= 1'b0;
            if (memio_read_req_valid && memio_read_req_ready) begin
                if (avgpool_busy) read_owner <= READ_OWNER_AVGPOOL;
                else if (tensor_busy) read_owner <= READ_OWNER_TENSOR;
                else if (bias_preload_busy) read_owner <= READ_OWNER_BIAS;
                else if (activation_busy) read_owner <= READ_OWNER_ACTIVATION;
                else if (rq_busy) read_owner <= READ_OWNER_REQUANT;
                else read_owner <= READ_OWNER_LOADER;
                read_owner_valid <= 1'b1;
            end
            if (memio_read_data_valid && memio_read_data_ready && memio_read_data_last) begin
                read_owner_valid <= 1'b0;
            end
            if (op_state != OP_WRITE_C) begin
                write_complete_pending <= 1'b0;
            end else if (writer_done && write_array_idx == active_arrays - 1'b1) begin
                write_complete_pending <= 1'b1;
            end

            // Configuration validation belongs to gemm_controller at start.
            // Register reset values are intentionally not a fault while idle.
            if (controller_error || loader_error || writer_error || rq_error || activation_error || tensor_error || avgpool_error || bias_preload_error || memio_error) begin
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
            if (tensor_done) begin
                done_r <= 1'b1;
            end
            if (avgpool_done) begin
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
