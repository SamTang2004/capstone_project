`timescale 1ns/1ps
module bcuda_dual_int4_ddr_top (
    input  logic        aclk,
    input  logic        aresetn,

    input  logic [6:0]  s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [6:0]  s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
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
        LOAD_IDLE,
        LOAD_GROUP_BEGIN,
        LOAD_A_WAIT,
        LOAD_A_DRAIN,
        LOAD_B_WAIT,
        LOAD_B_DRAIN,
        LOAD_B_NEXT,
        BIAS_WAIT,
        BIAS_NEXT,
        LOAD_GROUP_COMMIT,
        LOAD_GROUP_DRAIN,
        LOAD_PREFETCH_HOLD,
        LOAD_A_ACK,
        LOAD_MATRIX_GROUP_WAIT
    } load_state_t;
    typedef enum logic [2:0] {
        READ_NONE, READ_MATRIX, READ_CONV, READ_BIAS,
        READ_REQUANT, READ_ACTIVATION, READ_TENSOR
    } read_owner_t;

    logic [31:0] ctrl_word;
    logic [31:0] status_word;
    logic [32*16-1:0] regs_flat;
    logic [32*16-1:0] diagnostics_flat;
    logic [31:0] diag_total_cycles;
    logic [31:0] diag_engine_cycles;
    logic [31:0] diag_compute_cycles;
    logic [31:0] diag_load_cycles;
    logic [31:0] diag_cache_wait_cycles;
    logic [31:0] diag_cache_ready_cycles;
    logic [31:0] diag_ddr_read_commands;
    logic [31:0] diag_ddr_read_beats;
    logic [31:0] diag_ddr_write_commands;
    logic [31:0] diag_ddr_stall_cycles;
    logic [31:0] diag_core_rows;
    logic [31:0] diag_output_stall_cycles;
    logic [31:0] diag_descriptor_count;
    logic [31:0] diag_cache_full_cycles;
    logic [2:0] diag_max_allocated;
    logic [2:0] diag_max_ready;
    logic ctrl_start_d;
    logic command_stage1;
    logic command_stage2;
    logic dispatch_pulse;
    logic done_r;
    logic error_r;
    logic [11:0] error_sources_r;
    logic completion_pending;

    wire [31:0] mmio_a_base = regs_flat[2*32 +: 32];
    wire [31:0] mmio_b_base = regs_flat[3*32 +: 32];
    wire [31:0] mmio_c_base = regs_flat[4*32 +: 32];
    wire [31:0] mmio_a_stride = regs_flat[5*32 +: 32];
    wire [31:0] mmio_b_stride = regs_flat[6*32 +: 32];
    wire [31:0] mmio_c_stride = regs_flat[7*32 +: 32];
    wire [31:0] mmio_m_full = regs_flat[8*32 +: 32];
    wire [15:0] mmio_m = mmio_m_full[15:0];
    wire [15:0] mmio_n = regs_flat[9*32 +: 16];
    wire [15:0] mmio_k = regs_flat[10*32 +: 16];
    wire [4:0] mmio_kernel_h = regs_flat[11*32 +: 5];
    wire [7:0] mmio_kernel_w = regs_flat[12*32 +: 8];
    wire [31:0] mmio_stride_pair = regs_flat[13*32 +: 32];
    wire [31:0] mmio_pad_pair = regs_flat[14*32 +: 32];
    wire [31:0] mmio_bias_base = regs_flat[15*32 +: 32];

    logic [31:0] a_base, b_base, c_base;
    logic [31:0] a_stride, b_stride, c_stride;
    logic [15:0] dim_m, dim_n, dim_k;
    logic [31:0] element_count;
    logic [4:0] kernel_h;
    logic [7:0] kernel_w;
    logic [7:0] stride_h, stride_w, pad_h, pad_w;
    logic [31:0] bias_base;
    logic [31:0] op_param0, op_param1;
    logic conv_mode, fused_output_mode, fusion_config_mode;
    logic signed [31:0] fusion_multiplier;
    logic [5:0] fusion_right_shift;
    logic signed [31:0] fusion_output_zero_point;
    logic fusion_activation_enable, fusion_leaky_mode;
    logic [2:0] fusion_leaky_shift;
    logic requantize_mode, activation_mode, activation_leaky_mode;
    logic maxpool_mode, upsample_mode, concat_mode, avgpool_mode;
    logic [31:0] conv_spatial;
    logic [15:0] engine_m, engine_n, engine_k;
    logic engine_start;
    logic engine_busy, engine_done, engine_error;
    logic engine_compute_active, engine_load_wait_active;
    logic load_a_request, load_b_request;
    logic external_load_done;
    logic buffer_group_valid;
    logic external_group_valid;
    logic external_group_ready;
    logic [2:0] external_group_row;
    logic [7:0] external_group_a_mask;
    logic [31:0] external_group_a_data;
    logic [7:0] external_group_b0_mask;
    logic [31:0] external_group_b0_data;
    logic [7:0] external_group_b1_mask;
    logic [31:0] external_group_b1_data;
    logic [2:0] external_group_row_lookahead;
    logic [7:0] external_group_a_mask_lookahead;
    logic [31:0] external_group_a_data_lookahead;
    logic [7:0] external_group_b0_mask_lookahead;
    logic [31:0] external_group_b0_data_lookahead;
    logic [7:0] external_group_b1_mask_lookahead;
    logic [31:0] external_group_b1_data_lookahead;
    logic source_load_valid;
    logic source_load_ready;
    logic source_load_array;
    logic source_load_matrix;
    logic [2:0] source_load_row;
    logic [7:0] source_load_mask;
    logic [31:0] source_load_data;
    logic group_begin_valid;
    logic group_begin_ready;
    logic group_commit_valid;
    logic group_commit_ready;
    logic group_stream_done;
    logic lookahead_group_done;
    logic [2:0] group_committed_count;
    logic lookahead_descriptor_valid;
    logic lookahead_descriptor_ready;
    logic lookahead_external_fill;
    logic [15:0] lookahead_descriptor_id;
    logic [31:0] lookahead_a_base, lookahead_b0_base, lookahead_b1_base;
    logic [15:0] lookahead_a_stride;
    logic [15:0] lookahead_b0_stride, lookahead_b1_stride;
    logic [7:0] lookahead_a_row_mask, lookahead_b_row_mask;
    logic [7:0] lookahead_a_lane_mask;
    logic [7:0] lookahead_b0_lane_mask, lookahead_b1_lane_mask;
    logic [2:0] lookahead_allocated_tiles, lookahead_ready_tiles;
    logic [7:0] lookahead_outstanding_words;
    logic lookahead_cache_full, lookahead_error;
    logic row_stream_start;
    logic row_stream_start_ready;
    logic row_stream_input_ready;
    logic row_stream_output_valid;
    logic row_stream_output_ready;
    logic [2:0] row_stream_output_row;
    logic [7:0] row_stream_a_mask;
    logic [31:0] row_stream_a_data;
    logic [7:0] row_stream_b0_mask;
    logic [31:0] row_stream_b0_data;
    logic [7:0] row_stream_b1_mask;
    logic [31:0] row_stream_b1_data;
    logic row_stream_done, row_stream_error;
    logic [4:0] row_stream_customer0_occupancy;
    logic [4:0] row_stream_customer1_occupancy;
    logic [15:0] tile_m0, tile_n0, tile_k0;
    logic [3:0] tile_m_len, tile_k_len;
    logic [4:0] tile_n_cols;
    logic [1:0] active_arrays;
    logic prefetch_mode;
    logic [15:0] prefetch_m0, prefetch_n0, prefetch_k0;
    logic [3:0] prefetch_m_len, prefetch_k_len;
    logic [4:0] prefetch_n_cols;
    logic [1:0] prefetch_active_arrays;
    logic core_prefetch_valid;
    logic matrix_producer_active;
    logic [15:0] matrix_producer_m0;
    logic [15:0] matrix_producer_n0;
    logic [15:0] matrix_producer_k0;
    logic [15:0] matrix_producer_id;
    logic [31:0] matrix_producer_a_origin;
    logic [31:0] matrix_producer_b_origin;
    logic [31:0] matrix_producer_a_addr;
    logic [31:0] matrix_producer_b_addr;
    logic load_request_serviced;
    logic engine_output_valid, engine_output_ready, engine_output_array;
    logic [5:0] engine_output_addr;
    logic signed [31:0] engine_output_data;

    load_state_t load_state;
    read_owner_t read_owner;
    logic load_array_idx;
    logic load_buffer_flush;
    logic load_buffer_flush_done;
    logic load_buffer_error;
    logic conv_loader_start;
    logic bias_start;
    logic [3:0] current_b_cols;

    logic matrix_read_req_valid, matrix_read_req_ready;
    logic [31:0] matrix_read_req_addr;
    logic [4:0] matrix_read_req_len;
    logic matrix_read_word_valid, matrix_read_word_ready, matrix_read_word_last;
    logic [31:0] matrix_read_word_data;
    logic [3:0] matrix_read_word_keep;

    logic conv_loader_done, conv_loader_error;
    logic conv_loader_busy;
    logic conv_read_req_valid, conv_read_req_ready;
    logic [31:0] conv_read_req_addr;
    logic [4:0] conv_read_req_len;
    logic conv_read_data_valid, conv_read_data_ready, conv_read_data_last;
    logic [7:0] conv_read_data;
    logic conv_cache_we, conv_cache_matrix;
    logic [5:0] conv_cache_addr;
    logic signed [3:0] conv_cache_data;

    logic bias_done, bias_error;
    logic bias_read_req_valid, bias_read_req_ready;
    logic [31:0] bias_read_req_addr;
    logic bias_read_data_valid, bias_read_data_ready;
    logic [7:0] bias_read_data;
    logic bias_seed_valid, bias_seed_ready;
    logic [7:0] bias_seed_addr;
    logic signed [31:0] bias_seed_data;

    logic memio_read_req_valid, memio_read_req_ready;
    logic [31:0] memio_read_req_addr;
    logic [4:0] memio_read_req_len;
    logic memio_read_req_word_mode;
    logic memio_read_data_valid, memio_read_data_ready, memio_read_data_last;
    logic [7:0] memio_read_data;
    logic memio_read_word_valid, memio_read_word_ready, memio_read_word_last;
    logic [31:0] memio_read_word_data;
    logic [3:0] memio_read_word_keep;
    logic memio_write_req_ready;
    logic memio_busy, memio_error;

    logic rq_start, rq_busy, rq_done, rq_error;
    logic rq_read_req_valid, rq_read_req_ready;
    logic [31:0] rq_read_req_addr;
    logic rq_read_data_valid, rq_read_data_ready;
    logic [7:0] rq_read_data;
    logic rq_write_valid, rq_write_ready;
    logic [31:0] rq_write_addr, rq_write_data;
    logic [3:0] rq_write_strb;

    logic activation_start, activation_busy, activation_done, activation_error;
    logic activation_read_req_valid, activation_read_req_ready;
    logic [31:0] activation_read_req_addr;
    logic activation_read_data_valid, activation_read_data_ready;
    logic [7:0] activation_read_data;
    logic activation_write_valid, activation_write_ready;
    logic [31:0] activation_write_addr, activation_write_data;
    logic [3:0] activation_write_strb;

    logic tensor_start, tensor_busy, tensor_done, tensor_error;
    logic tensor_read_req_valid, tensor_read_req_ready;
    logic [31:0] tensor_read_req_addr;
    logic tensor_read_data_valid, tensor_read_data_ready;
    logic [7:0] tensor_read_data;
    logic tensor_write_valid, tensor_write_ready;
    logic [31:0] tensor_write_addr, tensor_write_data;
    logic [3:0] tensor_write_strb;

    logic selected_write_valid, selected_write_ready, selected_write_last;
    logic [31:0] selected_write_addr, selected_write_data;
    logic [3:0] selected_write_strb;
    logic queued_write_valid, queued_write_ready, queued_write_last;
    logic [31:0] queued_write_addr, queued_write_data;
    logic [3:0] queued_write_strb;
    logic [1:0] queued_write_occupancy;

    logic writer_input_last;
    logic writer_write_valid, writer_write_ready, writer_write_last;
    logic [31:0] writer_write_addr, writer_write_data;
    logic [3:0] writer_write_strb;
    logic legacy_writer_input_ready;
    logic legacy_writer_write_valid, legacy_writer_write_ready;
    logic legacy_writer_write_last;
    logic [31:0] legacy_writer_write_addr, legacy_writer_write_data;
    logic [3:0] legacy_writer_write_strb;
    logic fused_writer_input_ready;
    logic fused_writer_write_valid, fused_writer_write_ready;
    logic fused_writer_write_last;
    logic [31:0] fused_writer_write_addr, fused_writer_write_data;
    logic [3:0] fused_writer_write_strb;
    logic engine_seed_ready;
    logic writer_input_ready;
    logic packed_vector_valid, packed_vector_ready;
    logic packed_vector_array, packed_vector_matrix;
    logic [2:0] packed_vector_row;
    logic [7:0] packed_vector_mask;
    logic [31:0] packed_vector_data;
    logic [15:0] weight_cache_sequence_r;
    logic [7:0] weight_cache_query_index;
    logic weight_cache_hit;
    logic weight_cache_reuse;
    logic weight_cache_capture;
    logic weight_cache_replay_start;
    logic weight_cache_replay_busy;
    logic weight_cache_replay_done;
    logic weight_cache_replay_error;
    logic weight_cache_replay_valid;
    logic weight_cache_replay_ready;
    logic weight_cache_replay_array;
    logic weight_cache_replay_matrix;
    logic [2:0] weight_cache_replay_row;
    logic [7:0] weight_cache_replay_mask;
    logic [31:0] weight_cache_replay_data;
    logic [31:0] diag_weight_cache_hits;
    logic [31:0] diag_weight_cache_misses;

    wire start_pulse = ctrl_word[0] && !ctrl_start_d;
    wire operator_mode = requantize_mode || activation_mode ||
                         maxpool_mode || upsample_mode ||
                         concat_mode || avgpool_mode || fusion_config_mode;
    wire [3:0] requested_mode_count =
        {3'd0, ctrl_word[1]} + {3'd0, ctrl_word[2]} +
        {3'd0, ctrl_word[3]} + {3'd0, ctrl_word[5]} +
        {3'd0, ctrl_word[6]} + {3'd0, ctrl_word[7]} +
        {3'd0, ctrl_word[9]} + {3'd0, ctrl_word[11]};
    wire invalid_mode = requested_mode_count > 1 || !ctrl_word[8] ||
                        (ctrl_word[10] && !ctrl_word[2]);
    wire command_active = command_stage1 || command_stage2 || dispatch_pulse ||
                          engine_busy || load_state != LOAD_IDLE ||
                          rq_busy || activation_busy || tensor_busy ||
                          selected_write_valid || queued_write_valid ||
                          memio_busy || completion_pending;
    wire selected_loader_error =
        conv_loader_error || lookahead_error || row_stream_error ||
        weight_cache_replay_error;
    wire [3:0] array0_cols = tile_n_cols > 8 ? 4'd8 : tile_n_cols[3:0];
    wire [3:0] array1_cols = tile_n_cols > 8 ? tile_n_cols - 8 : 4'd0;
    wire [15:0] load_m0 = prefetch_mode ? prefetch_m0 : tile_m0;
    wire [15:0] load_n0 = prefetch_mode ? prefetch_n0 : tile_n0;
    wire [15:0] load_k0 = prefetch_mode ? prefetch_k0 : tile_k0;
    wire [3:0] load_m_len = prefetch_mode ? prefetch_m_len : tile_m_len;
    wire [3:0] load_k_len = prefetch_mode ? prefetch_k_len : tile_k_len;
    wire [4:0] load_n_cols = prefetch_mode ?
                             prefetch_n_cols : tile_n_cols;
    wire [1:0] load_active_arrays = prefetch_mode ?
                                    prefetch_active_arrays : active_arrays;
    wire [15:0] next_m0_calc =
        load_k0 + 16'd8 < engine_k ? load_m0 :
        load_n0 + 16'd16 < engine_n ? load_m0 :
        load_m0 + 16'd8 < engine_m ? load_m0 + 16'd8 : load_m0;
    wire [15:0] next_n0_calc =
        load_k0 + 16'd8 < engine_k ? load_n0 :
        load_n0 + 16'd16 < engine_n ? load_n0 + 16'd16 : 16'd0;
    wire [15:0] next_k0_calc =
        load_k0 + 16'd8 < engine_k ? load_k0 + 16'd8 : 16'd0;
    wire has_next_tile =
        load_k0 + 16'd8 < engine_k ||
        load_n0 + 16'd16 < engine_n ||
        load_m0 + 16'd8 < engine_m;
    wire [15:0] next_m_remaining = engine_m - next_m0_calc;
    wire [15:0] next_n_remaining = engine_n - next_n0_calc;
    wire [15:0] next_k_remaining = engine_k - next_k0_calc;
    wire [3:0] next_m_len_calc =
        next_m_remaining > 8 ? 4'd8 : next_m_remaining[3:0];
    wire [3:0] next_k_len_calc =
        next_k_remaining > 8 ? 4'd8 : next_k_remaining[3:0];
    wire [4:0] next_n_cols_calc =
        next_n_remaining > 16 ? 5'd16 : next_n_remaining[4:0];
    wire [1:0] next_active_arrays_calc =
        next_n_remaining > 8 ? 2'd2 : 2'd1;
    // The compute core has one A/B register set. Keep speculative DDR work in
    // the credited BRAM look-ahead cache until the controller requests it;
    // chaining a second direct-to-core preload can overwrite an untagged set.
    wire can_prefetch_next = 1'b0;

    wire [15:0] producer_m_remaining =
        engine_m - matrix_producer_m0;
    wire [15:0] producer_n_remaining =
        engine_n - matrix_producer_n0;
    wire [15:0] producer_k_remaining =
        engine_k - matrix_producer_k0;
    wire [3:0] producer_m_len =
        producer_m_remaining > 8 ? 4'd8 :
        producer_m_remaining[3:0];
    wire [3:0] producer_k_len =
        producer_k_remaining > 8 ? 4'd8 :
        producer_k_remaining[3:0];
    wire [4:0] producer_n_cols =
        producer_n_remaining > 16 ? 5'd16 :
        producer_n_remaining[4:0];

    function automatic logic [7:0] low_mask8(input logic [4:0] lanes);
        if (lanes >= 8)
            low_mask8 = 8'hff;
        else if (lanes == 0)
            low_mask8 = 8'h00;
        else
            low_mask8 = (9'h001 << lanes) - 1'b1;
    endfunction

    assign engine_start = dispatch_pulse && !operator_mode && !invalid_mode;
    assign rq_start = dispatch_pulse && requantize_mode && !invalid_mode;
    assign activation_start = dispatch_pulse && activation_mode && !invalid_mode;
    assign tensor_start = dispatch_pulse &&
                          (maxpool_mode || upsample_mode ||
                           concat_mode || avgpool_mode) && !invalid_mode;
    assign current_b_cols = load_array_idx ?
                            (load_n_cols > 8 ? load_n_cols - 8 : 4'd0) :
                            (load_n_cols > 8 ? 4'd8 : load_n_cols[3:0]);
    assign weight_cache_query_index =
        (load_n0 == 0 && load_k0 == 0) ? 8'd0 :
        weight_cache_sequence_r[7:0];
    assign weight_cache_reuse =
        conv_mode && load_m0 != 0 &&
        weight_cache_sequence_r < 16'd256 && weight_cache_hit;
    assign weight_cache_capture =
        conv_mode && load_m0 == 0 &&
        weight_cache_sequence_r < 16'd256;
    assign weight_cache_replay_start =
        conv_loader_start && weight_cache_reuse;

    assign source_load_valid =
        weight_cache_replay_valid || packed_vector_valid;
    assign source_load_array = weight_cache_replay_valid ?
                               weight_cache_replay_array :
                               packed_vector_array;
    assign source_load_matrix = weight_cache_replay_valid ?
                                weight_cache_replay_matrix :
                                packed_vector_matrix;
    assign source_load_row = weight_cache_replay_valid ?
                             weight_cache_replay_row :
                             packed_vector_row;
    assign source_load_mask = weight_cache_replay_valid ?
                              weight_cache_replay_mask :
                              packed_vector_mask;
    assign source_load_data = weight_cache_replay_valid ?
                              weight_cache_replay_data :
                              packed_vector_data;
    assign external_group_valid =
        (conv_mode ? row_stream_output_valid : buffer_group_valid) &&
        load_a_request && !core_prefetch_valid;
    assign external_group_row =
        conv_mode ? row_stream_output_row : external_group_row_lookahead;
    assign external_group_a_mask =
        conv_mode ? row_stream_a_mask : external_group_a_mask_lookahead;
    assign external_group_a_data =
        conv_mode ? row_stream_a_data : external_group_a_data_lookahead;
    assign external_group_b0_mask =
        conv_mode ? row_stream_b0_mask : external_group_b0_mask_lookahead;
    assign external_group_b0_data =
        conv_mode ? row_stream_b0_data : external_group_b0_data_lookahead;
    assign external_group_b1_mask =
        conv_mode ? row_stream_b1_mask : external_group_b1_mask_lookahead;
    assign external_group_b1_data =
        conv_mode ? row_stream_b1_data : external_group_b1_data_lookahead;
    assign row_stream_output_ready =
        conv_mode && external_group_ready && load_a_request &&
        !core_prefetch_valid;
    assign packed_vector_ready =
        conv_mode && row_stream_input_ready && !weight_cache_replay_valid;
    assign weight_cache_replay_ready =
        conv_mode && row_stream_input_ready;
    assign load_buffer_flush_done = 1'b0;
    assign load_buffer_error = 1'b0;
    assign lookahead_descriptor_valid = !conv_mode && matrix_producer_active;
    assign lookahead_external_fill = 1'b0;
    assign lookahead_descriptor_id =
        conv_mode ? {load_m0[4:0], load_n0[5:0], load_k0[4:0]} :
                    matrix_producer_id;
    assign lookahead_a_base = conv_mode ? 32'd0 :
                              matrix_producer_a_addr;
    assign lookahead_b0_base = conv_mode ? 32'd0 :
                               matrix_producer_b_addr;
    assign lookahead_b1_base = conv_mode ? 32'd0 :
                               matrix_producer_b_addr + 4;
    assign lookahead_a_stride = conv_mode ? 16'd4 :
                                a_stride[15:0];
    assign lookahead_b0_stride = conv_mode ? 16'd4 :
                                 b_stride[15:0];
    assign lookahead_b1_stride = conv_mode ? 16'd4 :
                                 b_stride[15:0];
    assign lookahead_a_row_mask = low_mask8(
        conv_mode ? {1'b0, load_m_len} : {1'b0, producer_m_len});
    assign lookahead_b_row_mask = low_mask8(
        conv_mode ? {1'b0, load_k_len} : {1'b0, producer_k_len});
    assign lookahead_a_lane_mask = low_mask8(
        conv_mode ? {1'b0, load_k_len} : {1'b0, producer_k_len});
    assign lookahead_b0_lane_mask = low_mask8(
        conv_mode ? (load_n_cols > 8 ? 5'd8 : load_n_cols) :
        (producer_n_cols > 8 ? 5'd8 : producer_n_cols));
    assign lookahead_b1_lane_mask = low_mask8(
        conv_mode ?
        (load_n_cols > 8 ? load_n_cols - 8 : 5'd0) :
        (producer_n_cols > 8 ? producer_n_cols - 8 : 5'd0));
    assign group_begin_valid =
        conv_mode ? load_state == LOAD_GROUP_BEGIN :
                    lookahead_descriptor_valid;
    assign group_begin_ready =
        conv_mode ? row_stream_start_ready : lookahead_descriptor_ready;
    assign row_stream_start =
        conv_mode && load_state == LOAD_GROUP_BEGIN && group_begin_ready;
    assign group_commit_valid =
        !conv_mode && load_state == LOAD_GROUP_COMMIT;
    assign group_stream_done =
        conv_mode ? row_stream_done : lookahead_group_done;
    wire output_last_col = engine_output_array ?
                           engine_output_addr[2:0] == array1_cols - 1'b1 :
                           engine_output_addr[2:0] == array0_cols - 1'b1;
    assign writer_input_last = engine_output_addr[5:3] == tile_m_len - 1'b1 &&
                               output_last_col &&
                               (engine_output_array || active_arrays == 1);
    assign engine_output_ready = fused_output_mode ?
                                 fused_writer_input_ready :
                                 legacy_writer_input_ready;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            ctrl_start_d <= 1'b0;
            command_stage1 <= 1'b0;
            command_stage2 <= 1'b0;
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
            element_count <= '0;
            kernel_h <= '0;
            kernel_w <= '0;
            stride_h <= '0;
            stride_w <= '0;
            pad_h <= '0;
            pad_w <= '0;
            bias_base <= '0;
            op_param0 <= '0;
            op_param1 <= '0;
            conv_mode <= 1'b0;
            fused_output_mode <= 1'b0;
            fusion_config_mode <= 1'b0;
            fusion_multiplier <= '0;
            fusion_right_shift <= '0;
            fusion_output_zero_point <= '0;
            fusion_activation_enable <= 1'b0;
            fusion_leaky_mode <= 1'b0;
            fusion_leaky_shift <= '0;
            requantize_mode <= 1'b0;
            activation_mode <= 1'b0;
            activation_leaky_mode <= 1'b0;
            maxpool_mode <= 1'b0;
            upsample_mode <= 1'b0;
            concat_mode <= 1'b0;
            avgpool_mode <= 1'b0;
            conv_spatial <= '0;
            engine_m <= '0;
            engine_n <= '0;
            engine_k <= '0;
            done_r <= 1'b0;
            error_r <= 1'b0;
            error_sources_r <= '0;
            completion_pending <= 1'b0;
        end else begin
            ctrl_start_d <= ctrl_word[0];
            command_stage1 <= start_pulse;
            command_stage2 <= command_stage1;
            dispatch_pulse <= command_stage2;

            if (start_pulse) begin
                a_base <= mmio_a_base;
                b_base <= mmio_b_base;
                c_base <= mmio_c_base;
                a_stride <= mmio_a_stride;
                b_stride <= mmio_b_stride;
                c_stride <= mmio_c_stride;
                dim_m <= mmio_m;
                dim_n <= mmio_n;
                dim_k <= mmio_k;
                element_count <= mmio_m_full;
                kernel_h <= mmio_kernel_h;
                kernel_w <= mmio_kernel_w;
                stride_h <= mmio_stride_pair[7:0];
                stride_w <= mmio_stride_pair[15:8];
                pad_h <= mmio_pad_pair[7:0];
                pad_w <= mmio_pad_pair[15:8];
                bias_base <= mmio_bias_base;
                op_param0 <= mmio_stride_pair;
                op_param1 <= mmio_pad_pair;
                conv_mode <= ctrl_word[2];
                fused_output_mode <= ctrl_word[10];
                fusion_config_mode <= ctrl_word[11];
                requantize_mode <= ctrl_word[1];
                activation_mode <= ctrl_word[3];
                activation_leaky_mode <= ctrl_word[4];
                maxpool_mode <= ctrl_word[5];
                upsample_mode <= ctrl_word[6];
                concat_mode <= ctrl_word[7];
                avgpool_mode <= ctrl_word[9];
                if (ctrl_word[11]) begin
                    fusion_multiplier <= $signed(mmio_stride_pair);
                    fusion_right_shift <= mmio_pad_pair[5:0];
                    fusion_leaky_shift <= mmio_pad_pair[10:8];
                    fusion_output_zero_point <= $signed(mmio_bias_base);
                    fusion_activation_enable <= ctrl_word[12];
                    fusion_leaky_mode <= ctrl_word[4];
                end
                done_r <= 1'b0;
                error_r <= invalid_mode;
                error_sources_r <= {invalid_mode, 11'd0};
            end else begin
                error_sources_r <= error_sources_r | {
                    1'b0,
                    lookahead_error,
                    load_buffer_error,
                    memio_error,
                    tensor_error,
                    activation_error,
                    rq_error,
                    bias_error,
                    row_stream_error,
                    conv_loader_error,
                    selected_loader_error,
                    engine_error
                };
            end
            if (command_stage1)
                conv_spatial <= dim_n * dim_k;
            if (command_stage2) begin
                engine_m <= conv_mode ? conv_spatial[15:0] : dim_m;
                engine_n <= conv_mode ? dim_m : dim_n;
                engine_k <= conv_mode ? a_stride[15:0] * kernel_h * kernel_w : dim_k;
            end

            if (engine_done || rq_done || activation_done || tensor_done ||
                (dispatch_pulse && fusion_config_mode))
                completion_pending <= 1'b1;
            if (completion_pending && !memio_busy &&
                !selected_write_valid && !queued_write_valid) begin
                completion_pending <= 1'b0;
                done_r <= 1'b1;
            end
            if (engine_error || selected_loader_error || bias_error ||
                rq_error || activation_error || tensor_error || memio_error ||
                load_buffer_error)
                error_r <= 1'b1;
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            matrix_producer_active <= 1'b0;
            matrix_producer_m0 <= '0;
            matrix_producer_n0 <= '0;
            matrix_producer_k0 <= '0;
            matrix_producer_id <= '0;
            matrix_producer_a_origin <= '0;
            matrix_producer_b_origin <= '0;
            matrix_producer_a_addr <= '0;
            matrix_producer_b_addr <= '0;
        end else if (engine_start) begin
            matrix_producer_active <= !conv_mode;
            matrix_producer_m0 <= '0;
            matrix_producer_n0 <= '0;
            matrix_producer_k0 <= '0;
            matrix_producer_id <= '0;
            matrix_producer_a_origin <= a_base;
            matrix_producer_b_origin <= b_base;
            matrix_producer_a_addr <= a_base;
            matrix_producer_b_addr <= b_base;
        end else if (!conv_mode && lookahead_descriptor_valid &&
                     lookahead_descriptor_ready) begin
            matrix_producer_id <= matrix_producer_id + 1'b1;
            if (matrix_producer_k0 + 16'd8 < engine_k) begin
                matrix_producer_k0 <= matrix_producer_k0 + 16'd8;
                matrix_producer_a_addr <=
                    matrix_producer_a_addr + 4;
                matrix_producer_b_addr <=
                    matrix_producer_b_addr + (b_stride << 3);
            end else if (matrix_producer_n0 + 16'd16 < engine_n) begin
                matrix_producer_n0 <= matrix_producer_n0 + 16'd16;
                matrix_producer_k0 <= '0;
                matrix_producer_b_origin <=
                    matrix_producer_b_origin + 8;
                matrix_producer_a_addr <= matrix_producer_a_origin;
                matrix_producer_b_addr <=
                    matrix_producer_b_origin + 8;
            end else if (matrix_producer_m0 + 16'd8 < engine_m) begin
                matrix_producer_m0 <= matrix_producer_m0 + 16'd8;
                matrix_producer_n0 <= '0;
                matrix_producer_k0 <= '0;
                matrix_producer_a_origin <=
                    matrix_producer_a_origin + (a_stride << 3);
                matrix_producer_b_origin <= b_base;
                matrix_producer_a_addr <=
                    matrix_producer_a_origin + (a_stride << 3);
                matrix_producer_b_addr <= b_base;
            end else begin
                matrix_producer_active <= 1'b0;
            end
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            load_state <= LOAD_IDLE;
            load_array_idx <= 1'b0;
            conv_loader_start <= 1'b0;
            bias_start <= 1'b0;
            load_buffer_flush <= 1'b0;
            external_load_done <= 1'b0;
            prefetch_mode <= 1'b0;
            prefetch_m0 <= '0;
            prefetch_n0 <= '0;
            prefetch_k0 <= '0;
            prefetch_m_len <= '0;
            prefetch_k_len <= '0;
            prefetch_n_cols <= '0;
            prefetch_active_arrays <= '0;
            core_prefetch_valid <= 1'b0;
            load_request_serviced <= 1'b0;
            weight_cache_sequence_r <= '0;
            diag_weight_cache_hits <= '0;
            diag_weight_cache_misses <= '0;
        end else begin
            conv_loader_start <= 1'b0;
            bias_start <= 1'b0;
            load_buffer_flush <= 1'b0;
            external_load_done <= 1'b0;

            if (engine_start) begin
                load_state <= LOAD_IDLE;
                prefetch_mode <= 1'b0;
                core_prefetch_valid <= 1'b0;
                load_request_serviced <= 1'b0;
                weight_cache_sequence_r <= '0;
                diag_weight_cache_hits <= '0;
                diag_weight_cache_misses <= '0;
            end else begin
                if (!load_a_request)
                    load_request_serviced <= 1'b0;
                if (load_a_request && core_prefetch_valid &&
                    !load_request_serviced) begin
                    external_load_done <= 1'b1;
                    core_prefetch_valid <= 1'b0;
                    load_request_serviced <= 1'b1;
                    if (can_prefetch_next) begin
                        prefetch_mode <= 1'b1;
                        prefetch_m0 <= next_m0_calc;
                        prefetch_n0 <= next_n0_calc;
                        prefetch_k0 <= next_k0_calc;
                        prefetch_m_len <= next_m_len_calc;
                        prefetch_k_len <= next_k_len_calc;
                        prefetch_n_cols <= next_n_cols_calc;
                        prefetch_active_arrays <= next_active_arrays_calc;
                        load_state <= conv_mode ?
                                      LOAD_GROUP_BEGIN :
                                      LOAD_PREFETCH_HOLD;
                    end else begin
                        prefetch_mode <= 1'b0;
                        load_state <= LOAD_IDLE;
                    end
                end else begin
                unique case (load_state)
                LOAD_IDLE: begin
                    if (load_a_request && !load_request_serviced) begin
                        prefetch_mode <= 1'b0;
                        load_array_idx <= 1'b0;
                        load_state <= conv_mode ?
                                      LOAD_GROUP_BEGIN :
                                      LOAD_GROUP_DRAIN;
                    end
                end

                LOAD_GROUP_BEGIN: if (group_begin_ready) begin
                    conv_loader_start <= 1'b1;
                    if (weight_cache_reuse)
                        diag_weight_cache_hits <=
                            diag_weight_cache_hits + 1'b1;
                    else if (conv_mode && load_m0 != 0)
                        diag_weight_cache_misses <=
                            diag_weight_cache_misses + 1'b1;
                    load_array_idx <= 1'b0;
                    load_state <= LOAD_MATRIX_GROUP_WAIT;
                end

                LOAD_MATRIX_GROUP_WAIT: if (conv_loader_done) begin
                    weight_cache_sequence_r <=
                        weight_cache_query_index + 1'b1;
                    if (conv_mode && load_k0 == 0 && bias_base != 0) begin
                        load_array_idx <= 1'b0;
                        bias_start <= 1'b1;
                        load_state <= BIAS_WAIT;
                    end else begin
                        load_state <= LOAD_GROUP_DRAIN;
                    end
                end

                LOAD_A_WAIT: if (conv_loader_done) begin
                    if (conv_mode) begin
                        load_buffer_flush <= 1'b1;
                        load_state <= LOAD_A_DRAIN;
                    end else begin
                        load_array_idx <= 1'b0;
                        load_state <= LOAD_B_NEXT;
                    end
                end

                LOAD_A_DRAIN: if (load_buffer_flush_done) begin
                    load_array_idx <= 1'b0;
                    load_state <= LOAD_B_NEXT;
                end

                LOAD_B_WAIT: if (conv_loader_done) begin
                    if (conv_mode) begin
                        load_buffer_flush <= 1'b1;
                        load_state <= LOAD_B_DRAIN;
                    end else if (!load_array_idx && load_active_arrays == 2) begin
                        load_array_idx <= 1'b1;
                        load_state <= LOAD_B_NEXT;
                    end else if (conv_mode && load_k0 == 0 && bias_base != 0) begin
                        load_array_idx <= 1'b0;
                        bias_start <= 1'b1;
                        load_state <= BIAS_WAIT;
                    end else begin
                        load_state <= conv_mode ?
                                      LOAD_GROUP_DRAIN :
                                      LOAD_GROUP_COMMIT;
                    end
                end

                LOAD_B_DRAIN: if (load_buffer_flush_done) begin
                    if (!load_array_idx && load_active_arrays == 2) begin
                        load_array_idx <= 1'b1;
                        load_state <= LOAD_B_NEXT;
                    end else if (conv_mode && load_k0 == 0 && bias_base != 0) begin
                        load_array_idx <= 1'b0;
                        bias_start <= 1'b1;
                        load_state <= BIAS_WAIT;
                    end else begin
                        load_state <= conv_mode ?
                                      LOAD_GROUP_DRAIN :
                                      LOAD_GROUP_COMMIT;
                    end
                end

                LOAD_B_NEXT: begin
                    conv_loader_start <= 1'b1;
                    load_state <= LOAD_B_WAIT;
                end

                BIAS_WAIT: if (bias_done) begin
                    if (!load_array_idx && load_active_arrays == 2) begin
                        load_array_idx <= 1'b1;
                        load_state <= BIAS_NEXT;
                    end else begin
                        load_state <= conv_mode ?
                                      LOAD_GROUP_DRAIN :
                                      LOAD_GROUP_COMMIT;
                    end
                end

                BIAS_NEXT: begin
                    bias_start <= 1'b1;
                    load_state <= BIAS_WAIT;
                end

                LOAD_GROUP_COMMIT: if (group_commit_ready)
                    load_state <= LOAD_GROUP_DRAIN;

                LOAD_GROUP_DRAIN: if (group_stream_done) begin
                    if (prefetch_mode) begin
                        if (load_a_request && !load_request_serviced) begin
                            external_load_done <= 1'b1;
                            load_request_serviced <= 1'b1;
                        end else begin
                            core_prefetch_valid <= 1'b1;
                        end
                        load_state <= LOAD_IDLE;
                    end else begin
                        external_load_done <= 1'b1;
                        load_request_serviced <= 1'b1;
                        if (can_prefetch_next) begin
                            prefetch_mode <= 1'b1;
                            prefetch_m0 <= next_m0_calc;
                            prefetch_n0 <= next_n0_calc;
                            prefetch_k0 <= next_k0_calc;
                            prefetch_m_len <= next_m_len_calc;
                            prefetch_k_len <= next_k_len_calc;
                            prefetch_n_cols <= next_n_cols_calc;
                            prefetch_active_arrays <=
                                next_active_arrays_calc;
                            load_array_idx <= 1'b0;
                            load_state <= conv_mode ?
                                          LOAD_GROUP_BEGIN :
                                          LOAD_PREFETCH_HOLD;
                        end else begin
                            prefetch_mode <= 1'b0;
                            load_state <= LOAD_IDLE;
                        end
                    end
                end

                LOAD_PREFETCH_HOLD: begin
                    if (!load_a_request)
                        load_state <= LOAD_GROUP_DRAIN;
                end
                LOAD_A_ACK: load_state <= LOAD_IDLE;

                    default: load_state <= LOAD_IDLE;
                endcase
                end
            end
        end
    end

    always_comb begin
        read_owner = READ_NONE;
        if (rq_busy)
            read_owner = READ_REQUANT;
        else if (activation_busy)
            read_owner = READ_ACTIVATION;
        else if (tensor_busy)
            read_owner = READ_TENSOR;
        else if (bias_read_req_valid || load_state == BIAS_WAIT)
            read_owner = READ_BIAS;
        else if (conv_mode && (conv_read_req_valid || conv_loader_busy))
            read_owner = READ_CONV;
        else if (!conv_mode &&
                 (matrix_read_req_valid ||
                  lookahead_outstanding_words != 0))
            read_owner = READ_MATRIX;

        memio_read_req_valid = 1'b0;
        memio_read_req_addr = '0;
        memio_read_req_len = 5'd1;
        memio_read_req_word_mode = 1'b0;
        memio_read_data_ready = 1'b0;
        memio_read_word_ready = 1'b0;
        matrix_read_req_ready = 1'b0;
        conv_read_req_ready = 1'b0;
        bias_read_req_ready = 1'b0;
        rq_read_req_ready = 1'b0;
        activation_read_req_ready = 1'b0;
        tensor_read_req_ready = 1'b0;
        matrix_read_word_valid = 1'b0;
        conv_read_data_valid = 1'b0;
        bias_read_data_valid = 1'b0;
        rq_read_data_valid = 1'b0;
        activation_read_data_valid = 1'b0;
        tensor_read_data_valid = 1'b0;
        matrix_read_word_data = memio_read_word_data;
        matrix_read_word_keep = memio_read_word_keep;
        matrix_read_word_last = memio_read_word_last;
        conv_read_data = memio_read_data;
        bias_read_data = memio_read_data;
        rq_read_data = memio_read_data;
        activation_read_data = memio_read_data;
        tensor_read_data = memio_read_data;
        conv_read_data_last = memio_read_data_last;

        unique case (read_owner)
            READ_MATRIX: begin
                memio_read_req_valid = matrix_read_req_valid;
                memio_read_req_addr = matrix_read_req_addr;
                memio_read_req_len = matrix_read_req_len;
                memio_read_req_word_mode = 1'b1;
                matrix_read_req_ready = memio_read_req_ready;
                matrix_read_word_valid = memio_read_word_valid;
                memio_read_word_ready = matrix_read_word_ready;
            end
            READ_CONV: begin
                memio_read_req_valid = conv_read_req_valid;
                memio_read_req_addr = conv_read_req_addr;
                memio_read_req_len = conv_read_req_len;
                conv_read_req_ready = memio_read_req_ready;
                conv_read_data_valid = memio_read_data_valid;
                memio_read_data_ready = conv_read_data_ready;
            end
            READ_BIAS: begin
                memio_read_req_valid = bias_read_req_valid;
                memio_read_req_addr = bias_read_req_addr;
                bias_read_req_ready = memio_read_req_ready;
                bias_read_data_valid = memio_read_data_valid;
                memio_read_data_ready = bias_read_data_ready;
            end
            READ_REQUANT: begin
                memio_read_req_valid = rq_read_req_valid;
                memio_read_req_addr = rq_read_req_addr;
                rq_read_req_ready = memio_read_req_ready;
                rq_read_data_valid = memio_read_data_valid;
                memio_read_data_ready = rq_read_data_ready;
            end
            READ_ACTIVATION: begin
                memio_read_req_valid = activation_read_req_valid;
                memio_read_req_addr = activation_read_req_addr;
                activation_read_req_ready = memio_read_req_ready;
                activation_read_data_valid = memio_read_data_valid;
                memio_read_data_ready = activation_read_data_ready;
            end
            READ_TENSOR: begin
                memio_read_req_valid = tensor_read_req_valid;
                memio_read_req_addr = tensor_read_req_addr;
                tensor_read_req_ready = memio_read_req_ready;
                tensor_read_data_valid = memio_read_data_valid;
                memio_read_data_ready = tensor_read_data_ready;
            end
            default: begin end
        endcase
    end

    always_ff @(posedge aclk) begin
        if (!aresetn || dispatch_pulse) begin
            diag_total_cycles <= '0;
            diag_engine_cycles <= '0;
            diag_compute_cycles <= '0;
            diag_load_cycles <= '0;
            diag_cache_wait_cycles <= '0;
            diag_cache_ready_cycles <= '0;
            diag_ddr_read_commands <= '0;
            diag_ddr_read_beats <= '0;
            diag_ddr_write_commands <= '0;
            diag_ddr_stall_cycles <= '0;
            diag_core_rows <= '0;
            diag_output_stall_cycles <= '0;
            diag_descriptor_count <= '0;
            diag_cache_full_cycles <= '0;
            diag_max_allocated <= '0;
            diag_max_ready <= '0;
        end else if (command_active) begin
            diag_total_cycles <= diag_total_cycles + 1'b1;
            if (engine_busy)
                diag_engine_cycles <= diag_engine_cycles + 1'b1;
            if (engine_compute_active)
                diag_compute_cycles <= diag_compute_cycles + 1'b1;
            if (engine_load_wait_active)
                diag_load_cycles <= diag_load_cycles + 1'b1;
            if (engine_load_wait_active &&
                !(conv_mode ? row_stream_output_valid :
                              buffer_group_valid))
                diag_cache_wait_cycles <= diag_cache_wait_cycles + 1'b1;
            if (conv_mode ? row_stream_customer0_occupancy != 0 :
                            lookahead_ready_tiles != 0)
                diag_cache_ready_cycles <= diag_cache_ready_cycles + 1'b1;
            if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write)
                diag_ddr_read_commands <= diag_ddr_read_commands + 1'b1;
            if (mem_read_valid && mem_read_ready)
                diag_ddr_read_beats <= diag_ddr_read_beats + 1'b1;
            if (mem_cmd_valid && mem_cmd_ready && mem_cmd_write)
                diag_ddr_write_commands <= diag_ddr_write_commands + 1'b1;
            if ((mem_cmd_valid && !mem_cmd_ready) ||
                (mem_read_valid && !mem_read_ready) ||
                (mem_write_valid && !mem_write_ready))
                diag_ddr_stall_cycles <= diag_ddr_stall_cycles + 1'b1;
            if (external_group_valid && external_group_ready &&
                load_a_request && !core_prefetch_valid)
                diag_core_rows <= diag_core_rows + 1'b1;
            if (engine_output_valid && !engine_output_ready)
                diag_output_stall_cycles <=
                    diag_output_stall_cycles + 1'b1;
            if ((lookahead_descriptor_valid && lookahead_descriptor_ready) ||
                row_stream_start)
                diag_descriptor_count <= diag_descriptor_count + 1'b1;
            if (lookahead_cache_full)
                diag_cache_full_cycles <= diag_cache_full_cycles + 1'b1;
            if (lookahead_allocated_tiles > diag_max_allocated)
                diag_max_allocated <= lookahead_allocated_tiles;
            if (lookahead_ready_tiles > diag_max_ready)
                diag_max_ready <= lookahead_ready_tiles;
        end
    end

    always_comb begin
        diagnostics_flat = '0;
        diagnostics_flat[0*32 +: 32] = diag_total_cycles;
        diagnostics_flat[1*32 +: 32] = diag_engine_cycles;
        diagnostics_flat[2*32 +: 32] = diag_compute_cycles;
        diagnostics_flat[3*32 +: 32] = diag_load_cycles;
        diagnostics_flat[4*32 +: 32] = diag_cache_wait_cycles;
        diagnostics_flat[5*32 +: 32] = diag_cache_ready_cycles;
        diagnostics_flat[6*32 +: 32] = diag_ddr_read_commands;
        diagnostics_flat[7*32 +: 32] = diag_ddr_read_beats;
        diagnostics_flat[8*32 +: 32] = diag_ddr_write_commands;
        diagnostics_flat[9*32 +: 32] = diag_ddr_stall_cycles;
        diagnostics_flat[10*32 +: 32] = diag_core_rows;
        diagnostics_flat[11*32 +: 32] = diag_output_stall_cycles;
        diagnostics_flat[12*32 +: 32] = diag_descriptor_count;
        diagnostics_flat[13*32 +: 32] = diag_cache_full_cycles;
        diagnostics_flat[14*32 +: 32] = {
            5'd0, lookahead_ready_tiles,
            5'd0, lookahead_allocated_tiles,
            5'd0, diag_max_ready,
            5'd0, diag_max_allocated
        };
        diagnostics_flat[15*32 +: 32] = 32'h4449_4132;
    end

    axi_lite_control control (
        .aclk, .aresetn,
        .s_axi_awaddr, .s_axi_awvalid, .s_axi_awready,
        .s_axi_wdata, .s_axi_wstrb, .s_axi_wvalid, .s_axi_wready,
        .s_axi_bresp, .s_axi_bvalid, .s_axi_bready,
        .s_axi_araddr, .s_axi_arvalid, .s_axi_arready,
        .s_axi_rdata, .s_axi_rresp, .s_axi_rvalid, .s_axi_rready,
        .status_word, .diagnostics_flat, .ctrl_word, .regs_flat
    );

    lookahead_tile_loader_int4 #(
        .TILE_BANKS(4)
    ) tile_loader (
        .clk(aclk), .rst_n(aresetn && !engine_start),
        .descriptor_valid(lookahead_descriptor_valid),
        .descriptor_ready(lookahead_descriptor_ready),
        .descriptor_id(lookahead_descriptor_id),
        .descriptor_a_base(lookahead_a_base),
        .descriptor_b0_base(lookahead_b0_base),
        .descriptor_b1_base(lookahead_b1_base),
        .descriptor_a_stride(lookahead_a_stride),
        .descriptor_b0_stride(lookahead_b0_stride),
        .descriptor_b1_stride(lookahead_b1_stride),
        .descriptor_a_row_mask(lookahead_a_row_mask),
        .descriptor_b_row_mask(lookahead_b_row_mask),
        .descriptor_a_lane_mask(lookahead_a_lane_mask),
        .descriptor_b0_lane_mask(lookahead_b0_lane_mask),
        .descriptor_b1_lane_mask(lookahead_b1_lane_mask),
        .descriptor_external_fill(lookahead_external_fill),
        .external_vector_valid(1'b0),
        .external_vector_ready(source_load_ready),
        .external_vector_array(1'b0),
        .external_vector_matrix(1'b0),
        .external_vector_row(3'd0),
        .external_vector_data(32'd0),
        .external_commit_valid(group_commit_valid),
        .external_commit_ready(group_commit_ready),
        .read_req_valid(matrix_read_req_valid),
        .read_req_ready(matrix_read_req_ready),
        .read_req_addr(matrix_read_req_addr),
        .read_req_len(matrix_read_req_len),
        .read_word_valid(matrix_read_word_valid),
        .read_word_ready(matrix_read_word_ready),
        .read_word_data(matrix_read_word_data),
        .read_word_keep(matrix_read_word_keep),
        .read_word_last(matrix_read_word_last),
        .output_valid(buffer_group_valid),
        .output_ready(!conv_mode && external_group_ready && load_a_request &&
                      !core_prefetch_valid),
        .output_descriptor_id(),
        .output_row(external_group_row_lookahead),
        .output_a_mask(external_group_a_mask_lookahead),
        .output_a_data(external_group_a_data_lookahead),
        .output_b0_mask(external_group_b0_mask_lookahead),
        .output_b0_data(external_group_b0_data_lookahead),
        .output_b1_mask(external_group_b1_mask_lookahead),
        .output_b1_data(external_group_b1_data_lookahead),
        .group_done(lookahead_group_done),
        .allocated_tiles(lookahead_allocated_tiles),
        .ready_tiles(lookahead_ready_tiles),
        .outstanding_words(lookahead_outstanding_words),
        .cache_full(lookahead_cache_full),
        .error(lookahead_error)
    );
    assign group_committed_count =
        conv_mode ? row_stream_customer0_occupancy[2:0] :
                    lookahead_ready_tiles;

    int4_dual_array_row_stream_buffer conv_row_stream (
        .clk(aclk), .rst_n(aresetn && !engine_start),
        .start(row_stream_start), .start_ready(row_stream_start_ready),
        .active_arrays(load_active_arrays),
        .expected_a_rows(low_mask8({1'b0, load_m_len})),
        .expected_b_rows(low_mask8({1'b0, load_k_len})),
        .input_valid(source_load_valid),
        .input_ready(row_stream_input_ready),
        .input_array(source_load_array),
        .input_matrix(source_load_matrix),
        .input_row(source_load_row),
        .input_mask(source_load_mask),
        .input_data(source_load_data),
        .output_valid(row_stream_output_valid),
        .output_ready(row_stream_output_ready),
        .output_row(row_stream_output_row),
        .output_a_mask(row_stream_a_mask),
        .output_a_data(row_stream_a_data),
        .output_b0_mask(row_stream_b0_mask),
        .output_b0_data(row_stream_b0_data),
        .output_b1_mask(row_stream_b1_mask),
        .output_b1_data(row_stream_b1_data),
        .done(row_stream_done), .error(row_stream_error),
        .customer0_occupancy(row_stream_customer0_occupancy),
        .customer1_occupancy(row_stream_customer1_occupancy)
    );

    int4_conv_weight_tile_cache #(
        .TILE_COUNT(256), .INDEX_W(8)
    ) conv_weight_cache (
        .clk(aclk), .rst_n(aresetn), .clear(engine_start),
        .query_index(weight_cache_query_index),
        .query_hit(weight_cache_hit),
        .capture_start(conv_loader_start && weight_cache_capture),
        .capture_index(weight_cache_query_index),
        .capture_valid(packed_vector_valid && packed_vector_ready),
        .capture_ready(),
        .capture_array(packed_vector_array),
        .capture_matrix(packed_vector_matrix),
        .capture_row(packed_vector_row),
        .capture_mask(packed_vector_mask),
        .capture_data(packed_vector_data),
        .capture_finish(conv_loader_done),
        .replay_start(weight_cache_replay_start),
        .replay_index(weight_cache_query_index),
        .replay_active_arrays(load_active_arrays),
        .replay_busy(weight_cache_replay_busy),
        .replay_done(weight_cache_replay_done),
        .replay_error(weight_cache_replay_error),
        .replay_valid(weight_cache_replay_valid),
        .replay_ready(weight_cache_replay_ready),
        .replay_array(weight_cache_replay_array),
        .replay_matrix(weight_cache_replay_matrix),
        .replay_row(weight_cache_replay_row),
        .replay_mask(weight_cache_replay_mask),
        .replay_data(weight_cache_replay_data)
    );

    dual_int4_tiled_compute engine (
        .clk(aclk), .rst_n(aresetn), .start(engine_start),
        .M(engine_m), .N(engine_n), .K(engine_k),
        .busy(engine_busy), .done(engine_done), .error(engine_error),
        .compute_active(engine_compute_active),
        .load_wait_active(engine_load_wait_active),
        .load_a_request, .load_b_request, .external_load_done,
        .external_load_preloaded(core_prefetch_valid && !engine_start),
        .external_load_valid(1'b0), .external_load_ready(),
        .external_load_array(1'b0), .external_load_matrix(1'b0),
        .external_load_row(3'd0), .external_load_mask(8'd0),
        .external_load_data(32'd0),
        .external_group_valid, .external_group_ready,
        .external_group_row,
        .external_group_a_mask, .external_group_a_data,
        .external_group_b0_mask, .external_group_b0_data,
        .external_group_b1_mask, .external_group_b1_data,
        .seed_valid(bias_seed_valid), .seed_ready(engine_seed_ready),
        .seed_array(load_array_idx), .seed_addr(bias_seed_addr[5:0]),
        .seed_data(bias_seed_data),
        .preserve_accumulator_seed(conv_mode && bias_base != 0),
        .m0(tile_m0), .n0(tile_n0), .k0(tile_k0),
        .m_len(tile_m_len), .k_len(tile_k_len),
        .n_group_cols(tile_n_cols), .active_arrays,
        .output_valid(engine_output_valid), .output_ready(engine_output_ready),
        .output_array(engine_output_array), .output_addr(engine_output_addr),
        .output_data(engine_output_data)
    );

    conv2d_tile_group_loader_int4_ddr conv_loader (
        .clk(aclk), .rst_n(aresetn), .start(conv_loader_start),
        .skip_weight_streams(weight_cache_reuse),
        .input_base_addr(a_base), .weight_base_addr(b_base),
        .input_c(a_stride[15:0]), .input_h(b_stride[15:0]),
        .input_w(c_stride[15:0]), .out_c(dim_m), .out_w(dim_k),
        .kernel_h({3'd0, kernel_h}), .kernel_w,
        .stride_h, .stride_w, .pad_h, .pad_w,
        .m0(load_m0), .k0(load_k0),
        .n0(load_n0), .m_len(load_m_len), .k_len(load_k_len),
        .n_cols(load_n_cols), .active_arrays(load_active_arrays),
        .busy(conv_loader_busy), .done(conv_loader_done),
        .error(conv_loader_error),
        .read_req_valid(conv_read_req_valid),
        .read_req_ready(conv_read_req_ready),
        .read_req_addr(conv_read_req_addr), .read_req_len(conv_read_req_len),
        .read_data_valid(conv_read_data_valid),
        .read_data_ready(conv_read_data_ready), .read_data(conv_read_data),
        .read_data_last(conv_read_data_last),
        .vector_valid(packed_vector_valid),
        .vector_ready(packed_vector_ready),
        .vector_array(packed_vector_array),
        .vector_matrix(packed_vector_matrix),
        .vector_row(packed_vector_row),
        .vector_mask(packed_vector_mask),
        .vector_data(packed_vector_data)
    );

    conv_bias_preload_ddr #(.TILE_SIZE(8)) bias_loader (
        .clk(aclk), .rst_n(aresetn), .start(bias_start),
        .bias_base_addr(bias_base),
        .channel_offset(load_n0 + {load_array_idx, 3'b000}),
        .tile_rows({1'b0, load_m_len}), .tile_cols({1'b0, current_b_cols}),
        .busy(), .done(bias_done), .error(bias_error),
        .read_req_valid(bias_read_req_valid),
        .read_req_ready(bias_read_req_ready),
        .read_req_addr(bias_read_req_addr),
        .read_data_valid(bias_read_data_valid),
        .read_data_ready(bias_read_data_ready), .read_data(bias_read_data),
        .accum_valid(bias_seed_valid), .accum_ready(engine_seed_ready),
        .accum_addr(bias_seed_addr), .accum_data(bias_seed_data)
    );

    int32_result_stream_writer writer (
        .clk(aclk), .rst_n(aresetn),
        .input_valid(engine_output_valid && !fused_output_mode),
        .input_ready(legacy_writer_input_ready),
        .input_array(engine_output_array), .input_addr(engine_output_addr),
        .input_data(engine_output_data), .input_last(writer_input_last),
        .conv_mode, .output_base_addr(c_base), .row_stride_bytes(c_stride),
        .m0(tile_m0), .n0(tile_n0), .out_h(dim_n), .out_w(dim_k),
        .write_req_valid(legacy_writer_write_valid),
        .write_req_ready(legacy_writer_write_ready),
        .write_req_addr(legacy_writer_write_addr),
        .write_req_data(legacy_writer_write_data),
        .write_req_strb(legacy_writer_write_strb),
        .write_req_last(legacy_writer_write_last)
    );

    fused_conv_requant_int4_writer fused_writer (
        .clk(aclk), .rst_n(aresetn),
        .input_valid(engine_output_valid && fused_output_mode),
        .input_ready(fused_writer_input_ready),
        .input_array(engine_output_array), .input_addr(engine_output_addr),
        .input_data(engine_output_data), .input_last(writer_input_last),
        .output_base_addr(c_base),
        .m0(tile_m0), .n0(tile_n0),
        .m_len(tile_m_len), .n_cols(tile_n_cols),
        .out_h(dim_n), .out_w(dim_k),
        .multiplier(fusion_multiplier),
        .right_shift(fusion_right_shift),
        .output_zero_point(fusion_output_zero_point),
        .activation_enable(fusion_activation_enable),
        .leaky_mode(fusion_leaky_mode),
        .leaky_shift(fusion_leaky_shift),
        .write_req_valid(fused_writer_write_valid),
        .write_req_ready(fused_writer_write_ready),
        .write_req_addr(fused_writer_write_addr),
        .write_req_data(fused_writer_write_data),
        .write_req_strb(fused_writer_write_strb),
        .write_req_last(fused_writer_write_last)
    );

    requantize_int4_ddr requantizer (
        .clk(aclk), .rst_n(aresetn), .start(rq_start),
        .input_base_addr(a_base), .output_base_addr(c_base),
        .element_count(element_count),
        .multiplier($signed(op_param0)),
        .right_shift(op_param1[5:0]),
        .output_zero_point($signed(bias_base)),
        .busy(rq_busy), .done(rq_done), .error(rq_error),
        .read_req_valid(rq_read_req_valid),
        .read_req_ready(rq_read_req_ready),
        .read_req_addr(rq_read_req_addr),
        .read_data_valid(rq_read_data_valid),
        .read_data_ready(rq_read_data_ready), .read_data(rq_read_data),
        .write_req_valid(rq_write_valid),
        .write_req_ready(rq_write_ready),
        .write_req_addr(rq_write_addr), .write_req_data(rq_write_data),
        .write_req_strb(rq_write_strb)
    );

    activation_int4_ddr activation (
        .clk(aclk), .rst_n(aresetn), .start(activation_start),
        .leaky_mode(activation_leaky_mode),
        .leaky_shift(op_param1[2:0]),
        .input_base_addr(a_base), .output_base_addr(c_base),
        .element_count(element_count),
        .busy(activation_busy), .done(activation_done),
        .error(activation_error),
        .read_req_valid(activation_read_req_valid),
        .read_req_ready(activation_read_req_ready),
        .read_req_addr(activation_read_req_addr),
        .read_data_valid(activation_read_data_valid),
        .read_data_ready(activation_read_data_ready),
        .read_data(activation_read_data),
        .write_req_valid(activation_write_valid),
        .write_req_ready(activation_write_ready),
        .write_req_addr(activation_write_addr),
        .write_req_data(activation_write_data),
        .write_req_strb(activation_write_strb)
    );

    // Tensor-op overlay matches the legacy command ABI:
    // A_STRIDE=N, B_STRIDE=C0, C_STRIDE=H, M=W, N=out_H, K=out_W.
    // REG13/REG14 carry concat channels or pool stride/padding.
    cnn_tensor_ops_int4_ddr tensor_ops (
        .clk(aclk), .rst_n(aresetn), .start(tensor_start),
        .mode(maxpool_mode ? 3'd1 :
              (upsample_mode ? 3'd2 :
               (concat_mode ? 3'd3 : 3'd4))),
        .input0_base_addr(a_base), .input1_base_addr(b_base),
        .input2_base_addr(bias_base), .output_base_addr(c_base),
        .n(a_stride[15:0]), .channels0(b_stride[15:0]),
        .channels1(concat_mode ? op_param0[15:0] : 16'd0),
        .channels2(concat_mode ? op_param1[15:0] : 16'd0),
        .input_h(c_stride[15:0]), .input_w(dim_m),
        .output_h(dim_n), .output_w(dim_k),
        .kernel_h({3'd0, kernel_h}), .kernel_w,
        .stride_h, .stride_w, .pad_h, .pad_w,
        .busy(tensor_busy), .done(tensor_done), .error(tensor_error),
        .read_req_valid(tensor_read_req_valid),
        .read_req_ready(tensor_read_req_ready),
        .read_req_addr(tensor_read_req_addr),
        .read_data_valid(tensor_read_data_valid),
        .read_data_ready(tensor_read_data_ready),
        .read_data(tensor_read_data),
        .write_req_valid(tensor_write_valid),
        .write_req_ready(tensor_write_ready),
        .write_req_addr(tensor_write_addr),
        .write_req_data(tensor_write_data),
        .write_req_strb(tensor_write_strb)
    );

    always_comb begin
        writer_write_valid = fused_output_mode ?
                             fused_writer_write_valid :
                             legacy_writer_write_valid;
        writer_write_addr = fused_output_mode ?
                            fused_writer_write_addr :
                            legacy_writer_write_addr;
        writer_write_data = fused_output_mode ?
                            fused_writer_write_data :
                            legacy_writer_write_data;
        writer_write_strb = fused_output_mode ?
                            fused_writer_write_strb :
                            legacy_writer_write_strb;
        writer_write_last = fused_output_mode ?
                            fused_writer_write_last :
                            legacy_writer_write_last;
        selected_write_valid = writer_write_valid;
        selected_write_addr = writer_write_addr;
        selected_write_data = writer_write_data;
        selected_write_strb = writer_write_strb;
        selected_write_last = writer_write_last;
        writer_write_ready = !operator_mode && selected_write_ready;
        rq_write_ready = 1'b0;
        activation_write_ready = 1'b0;
        tensor_write_ready = 1'b0;
        if (rq_busy) begin
            selected_write_valid = rq_write_valid;
            selected_write_addr = rq_write_addr;
            selected_write_data = rq_write_data;
            selected_write_strb = rq_write_strb;
            selected_write_last = 1'b1;
            rq_write_ready = selected_write_ready;
            writer_write_ready = 1'b0;
        end else if (activation_busy) begin
            selected_write_valid = activation_write_valid;
            selected_write_addr = activation_write_addr;
            selected_write_data = activation_write_data;
            selected_write_strb = activation_write_strb;
            selected_write_last = 1'b1;
            activation_write_ready = selected_write_ready;
            writer_write_ready = 1'b0;
        end else if (tensor_busy) begin
            selected_write_valid = tensor_write_valid;
            selected_write_addr = tensor_write_addr;
            selected_write_data = tensor_write_data;
            selected_write_strb = tensor_write_strb;
            selected_write_last = 1'b1;
            tensor_write_ready = selected_write_ready;
            writer_write_ready = 1'b0;
        end
        legacy_writer_write_ready = !fused_output_mode &&
                                    writer_write_ready;
        fused_writer_write_ready = fused_output_mode &&
                                   writer_write_ready;
    end

    bcuda_write_command_fifo write_command_fifo (
        .clk(aclk), .rst_n(aresetn),
        .input_valid(selected_write_valid),
        .input_ready(selected_write_ready),
        .input_addr(selected_write_addr),
        .input_data(selected_write_data),
        .input_strb(selected_write_strb),
        .input_last(selected_write_last),
        .output_valid(queued_write_valid),
        .output_ready(queued_write_ready),
        .output_addr(queued_write_addr),
        .output_data(queued_write_data),
        .output_strb(queued_write_strb),
        .output_last(queued_write_last),
        .occupancy(queued_write_occupancy)
    );

    memory_io_driver memio (
        .clk(aclk), .rst_n(aresetn),
        .invalidate_cache(dispatch_pulse && !invalid_mode),
        .read_req_valid(memio_read_req_valid),
        .read_req_ready(memio_read_req_ready),
        .read_req_addr(memio_read_req_addr),
        .read_req_len(memio_read_req_len),
        .read_req_word_mode(memio_read_req_word_mode),
        .read_data_valid(memio_read_data_valid),
        .read_data_ready(memio_read_data_ready),
        .read_data(memio_read_data), .read_data_last(memio_read_data_last),
        .read_word_valid(memio_read_word_valid),
        .read_word_ready(memio_read_word_ready),
        .read_word_data(memio_read_word_data),
        .read_word_keep(memio_read_word_keep),
        .read_word_last(memio_read_word_last),
        .write_req_valid(queued_write_valid),
        .write_req_ready(queued_write_ready),
        .write_req_addr(queued_write_addr),
        .write_req_data(queued_write_data),
        .write_req_strb(queued_write_strb),
        .write_req_last(queued_write_last),
        .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write, .mem_cmd_addr,
        .mem_cmd_len, .mem_cmd_wdata, .mem_cmd_wstrb,
        .mem_write_valid, .mem_write_ready, .mem_write_data,
        .mem_write_strb, .mem_write_last,
        .mem_read_valid, .mem_read_ready, .mem_read_data, .mem_read_last,
        .busy(memio_busy), .error(memio_error)
    );

    assign status_word = {
        4'd0,
        error_sources_r,
        memio_read_data_valid,
        memio_read_data_ready,
        mem_cmd_ready,
        mem_cmd_valid,
        memio_busy,
        writer_write_valid,
        conv_loader_busy,
        (lookahead_allocated_tiles != 0 ||
         lookahead_outstanding_words != 0),
        engine_busy,
        load_state,
        error_r,
        done_r,
        command_active
    };
endmodule
