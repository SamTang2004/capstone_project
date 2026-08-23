set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "packed_w4_weight_reader.sv"]
synth_design -top packed_w4_weight_reader -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
report_utilization -file [file join $report_dir "packed_w4_reader_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $report_dir "packed_w4_reader_timing.rpt"]
puts "Packed W4 reader OOC synthesis completed."
