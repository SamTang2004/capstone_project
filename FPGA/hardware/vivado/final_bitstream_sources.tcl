# Canonical source list for the current deployable PL bitstream.
#
# Regression benches may still compile every RTL module in hardware/rtl, but
# project/bitstream creation should consume only this manifest so older
# development wrappers cannot accidentally become part of the final image.

set final_top bcuda_dual_int4_ddr_top
set final_sim_top tb_bcuda_dual_int4_ddr_top
set final_bitstream_name bcuda_dual_int4_ddr_top_standalone.bit

set final_rtl_files [list \
    [file join $hw_dir "rtl" "axi_lite_control.sv"] \
    [file join $hw_dir "rtl" "memory_io_driver.sv"] \
    [file join $hw_dir "rtl" "bcuda_write_command_fifo.sv"] \
    [file join $hw_dir "rtl" "pe_int4_mac.sv"] \
    [file join $hw_dir "rtl" "systolic_array_8x8_int4.sv"] \
    [file join $hw_dir "rtl" "dual_systolic_array_8x8_int4.sv"] \
    [file join $hw_dir "rtl" "gemm_dual_8x8_int4_core.sv"] \
    [file join $hw_dir "rtl" "gemm_controller_int4.sv"] \
    [file join $hw_dir "rtl" "accumulator_buffer.sv"] \
    [file join $hw_dir "rtl" "parallel_accumulator_buffer.sv"] \
    [file join $hw_dir "rtl" "int4_scalar_vector_buffer.sv"] \
    [file join $hw_dir "rtl" "int4_streaming_scalar_row_packer.sv"] \
    [file join $hw_dir "rtl" "int4_dual_array_row_stream_buffer.sv"] \
    [file join $hw_dir "rtl" "int4_conv_weight_tile_cache.sv"] \
    [file join $hw_dir "rtl" "dual_int4_tiled_compute.sv"] \
    [file join $hw_dir "rtl" "lookahead_tile_loader_int4.sv"] \
    [file join $hw_dir "rtl" "conv2d_tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "conv2d_tile_group_loader_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "conv_bias_preload_ddr.sv"] \
    [file join $hw_dir "rtl" "int32_result_stream_writer.sv"] \
    [file join $hw_dir "rtl" "requantize_int4_pipeline.sv"] \
    [file join $hw_dir "rtl" "fused_conv_requant_int4_writer.sv"] \
    [file join $hw_dir "rtl" "requantize_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "activation_int4_parallel.sv"] \
    [file join $hw_dir "rtl" "activation_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "cnn_tensor_ops_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "bcuda_dual_int4_ddr_top.sv"] \
]

set final_sim_files [list \
    [file join $hw_dir "sim" "tb_bcuda_dual_int4_ddr_top.sv"] \
]
