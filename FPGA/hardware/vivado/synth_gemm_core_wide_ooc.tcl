set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

read_verilog -sv [file join $hw_dir "rtl" "pe_int8_mac.sv"]
read_verilog -sv [file join $hw_dir "rtl" "systolic_array_16x16.sv"]
read_verilog -sv [file join $hw_dir "rtl" "gemm_16x16_core.sv"]

synth_design -top gemm_16x16_core -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]

report_utilization -hierarchical -file [file join $hw_dir "gemm_core_wide_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $hw_dir "gemm_core_wide_timing.rpt"]

puts "Wide banked GEMM core OOC synthesis completed."
