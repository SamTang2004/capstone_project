set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

# Platform-only regressions retained during the Transformer migration.
set rtl_files [list \
    [file join $hw_dir "rtl" "axi_lite_control.sv"] \
    [file join $hw_dir "rtl" "lut_mul32x16_pipeline.sv"] \
    [file join $hw_dir "rtl" "memory_io_driver.sv"] \
    [file join $hw_dir "rtl" "bcuda_transformer_ddr_top.sv"] \
    [file join $hw_dir "rtl" "memory_io_axi3_bridge.sv"] \
    [file join $hw_dir "rtl" "weight_vector_decode.sv"] \
    [file join $hw_dir "rtl" "pe_int8_mac.sv"] \
    [file join $hw_dir "rtl" "matrix_array_8x8.sv"] \
    [file join $hw_dir "rtl" "gemm_8x8_tile_core.sv"] \
    [file join $hw_dir "rtl" "gemm_requantize_writer.sv"] \
    [file join $hw_dir "rtl" "transformer_gemm_scheduler.sv"] \
    [file join $hw_dir "rtl" "packed_w4_weight_reader.sv"] \
    [file join $hw_dir "rtl" "simple_dual_port_bram.sv"] \
    [file join $hw_dir "rtl" "ping_pong_tile_buffer.sv"] \
    [file join $hw_dir "rtl" "w4a8_operand_buffers.sv"] \
    [file join $hw_dir "rtl" "transformer_debug_counters.sv"] \
    [file join $hw_dir "rtl" "transformer_control_status.sv"] \
    [file join $hw_dir "rtl" "residual_add_int8.sv"] \
    [file join $hw_dir "rtl" "iterative_unsigned_divider.sv"] \
    [file join $hw_dir "rtl" "bias_add_int32.sv"] \
    [file join $hw_dir "rtl" "gelu_int8.sv"] \
    [file join $hw_dir "rtl" "layernorm_int8.sv"] \
    [file join $hw_dir "rtl" "softmax_int8.sv"] \
    [file join $hw_dir "rtl" "rowwise_norm_softmax_engine.sv"] \
    [file join $hw_dir "rtl" "read_word_to_byte_adapter.sv"] \
    [file join $hw_dir "rtl" "requantize_ddr.sv"] \
    [file join $hw_dir "rtl" "requantize_pack_int8.sv"] \
    [file join $hw_dir "rtl" "activation_b_pack_int8.sv"] \
    [file join $hw_dir "rtl" "transformer_elementwise_ddr.sv"] \
    [file join $hw_dir "rtl" "transformer_rowwise_ddr.sv"] \
    [file join $hw_dir "rtl" "heatvit_token_scorer_4lane.sv"] \
    [file join $hw_dir "rtl" "heatvit_token_pruner_ddr.sv"] \
]
set sim_files [list \
    [file join $hw_dir "sim" "tb_memory_io_axi3_bridge_burst.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_driver_burst.sv"] \
    [file join $hw_dir "sim" "tb_memory_io_load_balance_stress.sv"] \
    [file join $hw_dir "sim" "tb_gemm_8x8_tile_core.sv"] \
    [file join $hw_dir "sim" "tb_gemm_requantize_writer.sv"] \
    [file join $hw_dir "sim" "tb_transformer_gemm_scheduler.sv"] \
    [file join $hw_dir "sim" "tb_packed_w4_weight_reader.sv"] \
    [file join $hw_dir "sim" "tb_w4a8_operand_buffers.sv"] \
    [file join $hw_dir "sim" "tb_transformer_debug_counters.sv"] \
    [file join $hw_dir "sim" "tb_residual_add_int8.sv"] \
    [file join $hw_dir "sim" "tb_bias_add_int32.sv"] \
    [file join $hw_dir "sim" "tb_gelu_int8.sv"] \
    [file join $hw_dir "sim" "tb_layernorm_int8.sv"] \
    [file join $hw_dir "sim" "tb_softmax_int8.sv"] \
    [file join $hw_dir "sim" "tb_rowwise_norm_softmax_engine.sv"] \
    [file join $hw_dir "sim" "tb_rowwise_norm_softmax_equivalence.sv"] \
    [file join $hw_dir "sim" "tb_pe_int8_mac.sv"] \
    [file join $hw_dir "sim" "tb_ps_pl_mmio_control.sv"] \
    [file join $hw_dir "sim" "tb_bcuda_transformer_ddr_top.sv"] \
    [file join $hw_dir "sim" "tb_read_word_to_byte_adapter.sv"] \
    [file join $hw_dir "sim" "tb_requantize_ddr.sv"] \
    [file join $hw_dir "sim" "tb_requantize_pack_int8.sv"] \
    [file join $hw_dir "sim" "tb_activation_b_pack_int8.sv"] \
    [file join $hw_dir "sim" "tb_transformer_elementwise_ddr.sv"] \
    [file join $hw_dir "sim" "tb_transformer_rowwise_ddr.sv"] \
    [file join $hw_dir "sim" "tb_heatvit_token_scorer_4lane.sv"] \
    [file join $hw_dir "sim" "tb_heatvit_token_pruner_ddr.sv"] \
]

if {[info exists ::env(BCUDA_XSIM_WORK)]} {
    set sim_work [file normalize $::env(BCUDA_XSIM_WORK)]
} else {
    set sim_work [file join $hw_dir "xsim_platform"]
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
foreach source [concat $rtl_files $sim_files] {
    lappend compile_cmd [file nativename $source]
}
set glbl_source [file normalize [file join $vivado_bin ".." "data" "verilog" "src" "glbl.v"]]
lappend compile_cmd [file nativename $glbl_source]
if {[catch {exec {*}$compile_cmd} output options]} {
    puts $output
    error "xvlog failed: $options"
}

set test_tops {
    tb_gemm_8x8_tile_core
    tb_gemm_requantize_writer
    tb_transformer_gemm_scheduler
    tb_packed_w4_weight_reader
    tb_w4a8_operand_buffers
    tb_transformer_debug_counters
    tb_residual_add_int8
    tb_bias_add_int32
    tb_gelu_int8
    tb_layernorm_int8
    tb_softmax_int8
    tb_rowwise_norm_softmax_engine
    tb_rowwise_norm_softmax_equivalence
    tb_pe_int8_mac
    tb_requantize_ddr
    tb_requantize_pack_int8
    tb_activation_b_pack_int8
    tb_transformer_elementwise_ddr
    tb_transformer_rowwise_ddr
    tb_heatvit_token_scorer_4lane
    tb_heatvit_token_pruner_ddr
    tb_memory_io_driver_burst
    tb_memory_io_load_balance_stress
    tb_read_word_to_byte_adapter
    tb_memory_io_axi3_bridge_burst
    tb_ps_pl_mmio_control
    tb_bcuda_transformer_ddr_top
}
if {[info exists ::env(BCUDA_TEST_TOPS)]} {
    set test_tops [split $::env(BCUDA_TEST_TOPS)]
}
foreach top $test_tops {
    set sim_name "${top}_sim"
    if {[catch {exec $xelab $top glbl "-L" xpm "-L" unisims_ver "--timescale" 1ns/1ps "-s" $sim_name} output options]} {
        puts $output
        error "xelab failed for $top: $options"
    }
    if {[catch {exec $xsim $sim_name "-runall"} output options]} {
        puts $output
        error "xsim failed for $top: $options"
    }
    puts $output
    if {![regexp {golden trace passed} $output]} {
        error "$top did not report a golden trace pass"
    }
}
puts "Transformer migration platform simulations completed."
