set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "simple_dual_port_bram.sv"]
read_verilog -sv [file join $hw_dir "rtl" "ping_pong_tile_buffer.sv"]
read_verilog -sv [file join $hw_dir "rtl" "w4a8_operand_buffers.sv"]
synth_design -top w4a8_operand_buffers -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
report_utilization -file [file join $report_dir "w4a8_operand_buffers_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $report_dir "w4a8_operand_buffers_timing.rpt"]
puts "W4A8 operand-buffer OOC synthesis completed."
