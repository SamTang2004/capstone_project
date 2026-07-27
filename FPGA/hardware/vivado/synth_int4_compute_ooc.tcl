set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "int4_compute_ooc"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "pe_int4_mac.sv"]
read_verilog -sv [file join $hw_dir "rtl" "systolic_array_8x8_int4.sv"]
read_verilog -sv [file join $hw_dir "rtl" "dual_systolic_array_8x8_int4.sv"]

synth_design \
    -top dual_systolic_array_8x8_int4 \
    -part xc7z020clg400-1 \
    -mode out_of_context \
    -flatten_hierarchy rebuilt

create_clock -name int4_compute_clk -period 10.000 [get_ports clk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "dual_8x8_int4_utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 20 \
    -file [file join $report_dir "dual_8x8_int4_timing.rpt"]

puts "INT4 OOC utilization: [file join $report_dir dual_8x8_int4_utilization.rpt]"
puts "INT4 OOC timing: [file join $report_dir dual_8x8_int4_timing.rpt]"
