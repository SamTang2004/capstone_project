set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "int4_conv_weight_cache_ooc"]
file mkdir $report_dir

read_verilog -sv \
    [file join $hw_dir "rtl" "int4_conv_weight_tile_cache.sv"]
synth_design -top int4_conv_weight_tile_cache \
    -part xc7z020clg400-1 -mode out_of_context
create_clock -name cache_clk -period 10.000 [get_ports clk]
report_utilization -hierarchical \
    -file [file join $report_dir "utilization.rpt"]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $report_dir "timing.rpt"]
