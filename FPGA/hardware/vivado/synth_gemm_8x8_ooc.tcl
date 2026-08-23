set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports"]
file mkdir $report_dir

read_verilog -sv [file join $hw_dir "rtl" "pe_int8_mac.sv"]
read_verilog -sv [file join $hw_dir "rtl" "weight_vector_decode.sv"]
read_verilog -sv [file join $hw_dir "rtl" "matrix_array_8x8.sv"]
read_verilog -sv [file join $hw_dir "rtl" "gemm_8x8_tile_core.sv"]
synth_design -top gemm_8x8_tile_core -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
report_utilization -file [file join $report_dir "gemm_8x8_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $report_dir "gemm_8x8_timing.rpt"]
puts "8x8 W8/W4 GEMM OOC synthesis completed."
