set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

# The production regression is explicit. Legacy INT8 RTL remains available
# for checkpoint testing but is not compiled into the active INT4 library.
source [file join $script_dir "final_bitstream_sources.tcl"]
set rtl_files [concat $final_rtl_files [list \
    [file join $hw_dir "rtl" "memory_io_axi_bridge.sv"] \
    [file join $hw_dir "rtl" "tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "tile_writer_int4_ddr.sv"] \
    [file join $hw_dir "rtl" "int4_output_postprocess.sv"] \
    [file join $hw_dir "rtl" "int4_cnn_tile_pipeline.sv"] \
]]
set rtl_files [lsort -unique $rtl_files]
set sim_files [list \
    [file join $hw_dir "sim" "tb_pe_int4_mac.sv"] \
    [file join $hw_dir "sim" "tb_systolic_array_8x8_int4.sv"] \
    [file join $hw_dir "sim" "tb_dual_systolic_array_8x8_int4.sv"] \
    [file join $hw_dir "sim" "tb_tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_matrix_tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_matrix_tile_group_loader_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_conv2d_tile_loader_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_conv2d_tile_group_loader_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_tile_writer_int4_ddr.sv"] \
    [file join $hw_dir "sim" "tb_gemm_controller_int4.sv"] \
    [file join $hw_dir "sim" "tb_gemm_dual_8x8_int4_core.sv"] \
    [file join $hw_dir "sim" "tb_int4_scalar_vector_buffer.sv"] \
    [file join $hw_dir "sim" "tb_int4_streaming_scalar_row_packer.sv"] \
    [file join $hw_dir "sim" "tb_int4_dual_array_row_stream_buffer.sv"] \
    [file join $hw_dir "sim" "tb_parallel_accumulator_buffer.sv"] \
    [file join $hw_dir "sim" "tb_int4_tile_group_buffer.sv"] \
    [file join $hw_dir "sim" "tb_int4_load_path_throughput.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_burst.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_multiread.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_multiline.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_strided_words.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_axi_bridge_burst.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_axi_bridge_outstanding.sv"] \
    [file join $hw_dir "sim" "tb_dual_int4_tiled_compute.sv"] \
    [file join $hw_dir "sim" "tb_activation_int4_parallel.sv"] \
    [file join $hw_dir "sim" "tb_requantize_int4_pipeline.sv"] \
    [file join $hw_dir "sim" "tb_fused_conv_requant_int4_writer.sv"] \
    [file join $hw_dir "sim" "tb_int4_output_postprocess.sv"] \
    [file join $hw_dir "sim" "tb_int4_cnn_tile_pipeline.sv"] \
    [file join $hw_dir "sim" "tb_bcuda_dual_int4_ddr_top.sv"] \
]

if {[info exists ::env(BCUDA_XSIM_WORK)]} {
    set sim_work [file normalize $::env(BCUDA_XSIM_WORK)]
} else {
    set sim_work [file join $hw_dir "xsim_golden"]
}
file mkdir $sim_work
cd $sim_work

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]

set compile_cmd [list $xvlog "--sv"]
foreach f [concat $rtl_files $sim_files] {
    lappend compile_cmd [file nativename $f]
}

puts "===== Compiling Budget-CUDA RTL and golden testbenches ====="
if {[catch {exec {*}$compile_cmd} compile_output compile_options]} {
    puts $compile_output
    error "xvlog failed: $compile_options"
}
puts $compile_output

# The active memory tests exercise cache-line reads and coalesced AXI bursts.
set test_tops {
    tb_pe_int4_mac
    tb_systolic_array_8x8_int4
    tb_dual_systolic_array_8x8_int4
    tb_tile_loader_int4_ddr
    tb_matrix_tile_loader_int4_ddr
    tb_matrix_tile_group_loader_int4_ddr
    tb_tile_writer_int4_ddr
    tb_conv2d_tile_loader_int4_ddr
    tb_conv2d_tile_group_loader_int4_ddr
    tb_gemm_controller_int4
    tb_gemm_dual_8x8_int4_core
    tb_int4_scalar_vector_buffer
    tb_int4_streaming_scalar_row_packer
    tb_int4_dual_array_row_stream_buffer
    tb_parallel_accumulator_buffer
    tb_int4_tile_group_buffer
    tb_int4_load_path_throughput
    tb_memory_io_driver_burst
    tb_memory_io_driver_multiread
    tb_memory_io_driver_multiline
    tb_memory_io_driver_strided_words
    tb_memory_io_axi_bridge_burst
    tb_memory_io_axi_bridge_outstanding
    tb_dual_int4_tiled_compute
    tb_activation_int4_parallel
    tb_requantize_int4_pipeline
    tb_fused_conv_requant_int4_writer
    tb_int4_output_postprocess
    tb_int4_cnn_tile_pipeline
    tb_bcuda_dual_int4_ddr_top
}
if {[info exists ::env(BCUDA_TEST_TOP)]} {
    set test_tops [list $::env(BCUDA_TEST_TOP)]
} elseif {$argc > 0} {
    set test_tops [list [lindex $argv 0]]
}

foreach top $test_tops {
    puts "===== Running $top ====="
    set sim_name "${top}_sim"

    if {[catch {exec $xelab $top "-s" $sim_name "-timescale" "1ns/1ps"} elab_output elab_options]} {
        puts $elab_output
        error "xelab failed for $top: $elab_options"
    }
    puts $elab_output

    if {[catch {exec $xsim $sim_name "-runall"} sim_output sim_options]} {
        puts $sim_output
        error "xsim failed for $top: $sim_options"
    }
    puts $sim_output

    if {![regexp {golden trace passed} $sim_output]} {
        error "$top did not report a golden trace pass"
    }
}

puts "All Budget-CUDA hardware simulations completed."
