# Standalone synthesis only. No board project, implementation or programming.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build work_packer_synth]
file mkdir $out
cd $out
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv [list [file join $root rtl llm_work_finder8.sv] \
    [file join $root rtl llm_survivor_buffer8.sv] [file join $root rtl llm_work_packer8.sv]]
synth_design -top llm_work_packer8 -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
set_clock_uncertainty 0.300 [get_clocks clk]
report_utilization -file utilization.rpt
report_timing_summary -file timing_synth.rpt
write_checkpoint -force work_packer_synth.dcp
puts "PASS: work packer standalone synthesis"
