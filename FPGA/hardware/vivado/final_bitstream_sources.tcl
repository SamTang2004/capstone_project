# Canonical source list for the current deployable PL bitstream.
#
# Every production build and regression source list derives from this manifest.

set final_top bcuda_tiled_systolic_ddr_top
set final_sim_top tb_bcuda_tiled_systolic_ddr_top
set final_bitstream_name bcuda_tiled_systolic_ddr_top_standalone.bit

set final_rtl_files [list \
    [file join $hw_dir "rtl" "axi_lite_control.sv"] \
    [file join $hw_dir "rtl" "memory_io_driver.sv"] \
    [file join $hw_dir "rtl" "tile_loader_ddr.sv"] \
    [file join $hw_dir "rtl" "tile_writer_ddr.sv"] \
    [file join $hw_dir "rtl" "lut_mul32x16_pipeline.sv"] \
    [file join $hw_dir "rtl" "conv2d_activation_address_gen.sv"] \
    [file join $hw_dir "rtl" "conv2d_tile_loader_ddr.sv"] \
    [file join $hw_dir "rtl" "conv2d_tile_writer_ddr.sv"] \
    [file join $hw_dir "rtl" "conv_bias_preload_ddr.sv"] \
    [file join $hw_dir "rtl" "requantize_ddr.sv"] \
    [file join $hw_dir "rtl" "pe_int8_mac.sv"] \
    [file join $hw_dir "rtl" "systolic_array_16x16.sv"] \
    [file join $hw_dir "rtl" "gemm_16x16_core.sv"] \
    [file join $hw_dir "rtl" "gemm_array_cluster_8.sv"] \
    [file join $hw_dir "rtl" "activation_unfold_parallel.sv"] \
    [file join $hw_dir "rtl" "activation_ddr_parallel.sv"] \
    [file join $hw_dir "rtl" "cnn_tensor_ops_ddr.sv"] \
    [file join $hw_dir "rtl" "avgpool_ddr.sv"] \
    [file join $hw_dir "rtl" "gemm_controller.sv"] \
    [file join $hw_dir "rtl" "accumulator_buffer.sv"] \
    [file join $hw_dir "rtl" "bcuda_tiled_systolic_ddr_top.sv"] \
]

set final_sim_files [list \
    [file join $hw_dir "sim" "tb_bcuda_tiled_systolic_ddr_top.sv"] \
]

set board_wrapper_rtl_files [list \
    [file join $hw_dir "rtl" "bcuda_tiled_systolic_ddr_ip.v"] \
    [file join $hw_dir "rtl" "memory_io_axi3_bridge.sv"] \
    [file join $hw_dir "rtl" "memory_io_axi3_bridge_ip.v"] \
]
