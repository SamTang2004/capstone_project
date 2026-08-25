set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set final_rtl_files [list \
    [file join $hw_dir "rtl" "axi_lite_control.sv"] \
    [file join $hw_dir "rtl" "pe_int8_mac.sv"] \
    [file join $hw_dir "rtl" "weight_vector_decode.sv"] \
    [file join $hw_dir "rtl" "matrix_array_8x8.sv"] \
    [file join $hw_dir "rtl" "gemm_8x8_tile_core.sv"] \
    [file join $hw_dir "rtl" "gemm_requantize_writer.sv"] \
    [file join $hw_dir "rtl" "transformer_gemm_scheduler.sv"] \
    [file join $hw_dir "rtl" "requantize_ddr.sv"] \
    [file join $hw_dir "rtl" "requantize_pack_int8.sv"] \
    [file join $hw_dir "rtl" "activation_b_pack_int8.sv"] \
    [file join $hw_dir "rtl" "gelu_int8.sv"] \
    [file join $hw_dir "rtl" "residual_add_int8.sv"] \
    [file join $hw_dir "rtl" "iterative_unsigned_divider.sv"] \
    [file join $hw_dir "rtl" "transformer_elementwise_ddr.sv"] \
    [file join $hw_dir "rtl" "layernorm_int8.sv"] \
    [file join $hw_dir "rtl" "softmax_int8.sv"] \
    [file join $hw_dir "rtl" "rowwise_norm_softmax_engine.sv"] \
    [file join $hw_dir "rtl" "transformer_rowwise_ddr.sv"] \
    [file join $hw_dir "rtl" "transformer_debug_counters.sv"] \
    [file join $hw_dir "rtl" "transformer_control_status.sv"] \
    [file join $hw_dir "rtl" "heatvit_token_scorer_4lane.sv"] \
    [file join $hw_dir "rtl" "heatvit_token_pruner_ddr.sv"] \
    [file join $hw_dir "rtl" "memory_io_driver.sv"] \
    [file join $hw_dir "rtl" "bcuda_transformer_ddr_top.sv"] \
]
set board_wrapper_rtl_files [list \
    [file join $hw_dir "rtl" "bcuda_transformer_ddr_ip.v"] \
    [file join $hw_dir "rtl" "memory_io_axi3_bridge.sv"] \
    [file join $hw_dir "rtl" "memory_io_axi3_bridge_ip.v"] \
]
