set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "int4_tile_group_buffer_ooc"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "int4_tile_group_buffer.sv"]

synth_design \
    -top int4_tile_group_buffer \
    -part xc7z020clg400-1 \
    -flatten_hierarchy rebuilt

create_clock -name tile_group_clk -period 10.000 [get_ports clk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 10 \
    -file [file join $report_dir "timing_summary.rpt"]
