set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set work_dir [file join $hw_dir "xsim_int4_top_compile"]
file mkdir $work_dir
cd $work_dir

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]

set sources [list \
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
    [file join $hw_dir "rtl" "int4_tile_group_buffer.sv"] \
    [file join $hw_dir "rtl" "lookahead_tile_loader_int4.sv"] \
    [file join $hw_dir "rtl" "dual_int4_tiled_compute.sv"] \
    [file join $hw_dir "rtl" "matrix_tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "matrix_tile_group_loader_int4_ddr.sv"] \
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
    [file join $hw_dir "sim" "tb_bcuda_dual_int4_ddr_top.sv"] \
]

set compile_cmd [list $xvlog "--sv"]
foreach source $sources { lappend compile_cmd [file nativename $source] }
if {[catch {exec {*}$compile_cmd} output options]} {
    puts $output
    error "INT4 top xvlog failed: $options"
}
puts $output

if {[catch {exec $xelab bcuda_dual_int4_ddr_top -s bcuda_dual_int4_ddr_top_compile -timescale 1ns/1ps} output options]} {
    puts $output
    error "INT4 top elaboration failed: $options"
}
puts $output
puts "Dual INT4 DDR top compilation passed."

if {[info exists ::env(BCUDA_RUN_INT4_TOP_SIM)]} {
    set xsim [file join $vivado_bin "xsim.bat"]
    if {[catch {exec $xelab tb_bcuda_dual_int4_ddr_top -s tb_bcuda_dual_int4_ddr_top_sim -timescale 1ns/1ps} output options]} {
        puts $output
        error "INT4 top testbench elaboration failed: $options"
    }
    puts $output
    if {[catch {exec $xsim tb_bcuda_dual_int4_ddr_top_sim -runall} output options]} {
        puts $output
        error "INT4 top simulation failed: $options"
    }
    puts $output
    if {[regexp {Fatal:} $output] ||
        ![regexp {Dual INT4 DDR top golden trace passed: upsample/concat} $output]} {
        error "INT4 top simulation did not reach the final golden trace pass"
    }
}
