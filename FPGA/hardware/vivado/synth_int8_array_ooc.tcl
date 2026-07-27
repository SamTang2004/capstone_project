set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "int4_compute_ooc"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "pe_int8_mac.sv"]
read_verilog -sv [file join $hw_dir "rtl" "systolic_array_16x16.sv"]

synth_design \
    -top systolic_array_16x16 \
    -part xc7z020clg400-1 \
    -mode out_of_context \
    -flatten_hierarchy rebuilt

create_clock -name int8_compute_clk -period 10.000 [get_ports clk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "single_16x16_int8_utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 20 \
    -file [file join $report_dir "single_16x16_int8_timing.rpt"]

puts "INT8 OOC utilization: [file join $report_dir single_16x16_int8_utilization.rpt]"
puts "INT8 OOC timing: [file join $report_dir single_16x16_int8_timing.rpt]"
