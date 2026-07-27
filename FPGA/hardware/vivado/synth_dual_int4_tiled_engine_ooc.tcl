set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "dual_int4_tiled_engine_ooc"]
file mkdir $report_dir

foreach source {
    pe_int4_mac.sv
    systolic_array_8x8_int4.sv
    dual_systolic_array_8x8_int4.sv
    gemm_controller_int4.sv
    accumulator_buffer.sv
    parallel_accumulator_buffer.sv
    gemm_dual_8x8_int4_core.sv
    int4_scalar_vector_buffer.sv
    dual_int4_tiled_compute.sv
} {
    read_verilog -sv [file join $hw_dir "rtl" $source]
}

synth_design \
    -top dual_int4_tiled_compute \
    -part xc7z020clg400-1 \
    -mode out_of_context \
    -flatten_hierarchy rebuilt

create_clock -name int4_engine_clk -period 10.000 [get_ports clk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 20 \
    -file [file join $report_dir "timing.rpt"]

puts "Dual INT4 tiled engine utilization: [file join $report_dir utilization.rpt]"
puts "Dual INT4 tiled engine timing: [file join $report_dir timing.rpt]"
