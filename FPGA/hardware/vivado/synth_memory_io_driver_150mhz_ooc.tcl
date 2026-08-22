set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

read_verilog -sv [file join $hw_dir "rtl" "memory_io_driver.sv"]
synth_design -top memory_io_driver -part xc7z020clg400-1 -mode out_of_context \
    -generic WIDE_RESPONSE_ENABLE=1 \
    -generic READ_AHEAD_ENABLE=1 \
    -generic READ_REQUEST_DEPTH=8
create_clock -name clk -period 6.667 [get_ports clk]

report_utilization -file [file join $hw_dir "memory_io_driver_150mhz_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $hw_dir "memory_io_driver_150mhz_timing.rpt"]

puts "Memory IO driver 150 MHz OOC synthesis completed."
