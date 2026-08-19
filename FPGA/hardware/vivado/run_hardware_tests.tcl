set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

# Keep regressions tied to the deployable source manifest. Prototype RTL must
# opt into a separate test flow instead of slowing or contaminating this one.
source [file join $script_dir "final_bitstream_sources.tcl"]
set rtl_files [concat $final_rtl_files $board_wrapper_rtl_files]
set sim_files [list \
    [file join $hw_dir "sim" "tb_pe_int8_mac.sv"] \
    [file join $hw_dir "sim" "tb_tile_loader_ddr.sv"] \
    [file join $hw_dir "sim" "tb_tile_writer_ddr.sv"] \
    [file join $hw_dir "sim" "tb_conv2d_tile_loader_ddr.sv"] \
    [file join $hw_dir "sim" "tb_conv_bias_preload_ddr.sv"] \
    [file join $hw_dir "sim" "tb_gemm_array_cluster_8.sv"] \
    [file join $hw_dir "sim" "tb_activation_unfold_parallel.sv"] \
    [file join $hw_dir "sim" "tb_activation_ddr_parallel.sv"] \
    [file join $hw_dir "sim" "tb_cnn_tensor_ops_ddr.sv"] \
    [file join $hw_dir "sim" "tb_avgpool_ddr.sv"] \
    [file join $hw_dir "sim" "tb_requantize_ddr.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_burst.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_axi3_bridge_burst.sv"] \
    [file join $hw_dir "sim" "tb_conv_writer_axi_path.sv"] \
    [file join $hw_dir "sim" "tb_ps_pl_mmio_control.sv"] \
    [file join $hw_dir "sim" "tb_bcuda_tiled_systolic_ddr_top.sv"] \
    [file join $hw_dir "sim" "tb_accumulator_buffer.sv"] \
    [file join $hw_dir "sim" "tb_gemm_controller.sv"] \
    [file join $hw_dir "sim" "tb_systolic_array_16x16.sv"] \
    [file join $hw_dir "sim" "tb_gemm_16x16_core.sv"] \
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
    tb_pe_int8_mac
    tb_tile_loader_ddr
    tb_tile_writer_ddr
    tb_conv2d_tile_loader_ddr
    tb_conv_bias_preload_ddr
    tb_gemm_array_cluster_8
    tb_activation_unfold_parallel
    tb_activation_ddr_parallel
    tb_cnn_tensor_ops_ddr
    tb_avgpool_ddr
    tb_requantize_ddr
    tb_memory_io_driver_burst
    tb_memory_io_axi3_bridge_burst
    tb_conv_writer_axi_path
    tb_ps_pl_mmio_control
    tb_bcuda_tiled_systolic_ddr_top
    tb_accumulator_buffer
    tb_gemm_controller
    tb_systolic_array_16x16
    tb_gemm_16x16_core
}

foreach top $test_tops {
    puts "===== Running $top ====="
    set sim_name "${top}_sim"

    if {[catch {exec $xelab $top "-s" $sim_name} elab_output elab_options]} {
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
