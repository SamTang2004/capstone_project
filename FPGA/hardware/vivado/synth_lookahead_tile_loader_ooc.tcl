set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "lookahead_tile_loader_ooc"]
file mkdir $report_dir

read_verilog -sv \
    [file join $hw_dir "rtl" "lookahead_tile_loader_int4.sv"]

synth_design \
    -top lookahead_tile_loader_int4 \
    -part xc7z020clg400-1 \
    -mode out_of_context \
    -flatten_hierarchy rebuilt

create_clock -name lookahead_loader_clk -period 10.000 [get_ports clk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 20 \
    -file [file join $report_dir "timing.rpt"]

puts "Look-ahead TileLoader utilization: [file join $report_dir utilization.rpt]"
puts "Look-ahead TileLoader timing: [file join $report_dir timing.rpt]"
