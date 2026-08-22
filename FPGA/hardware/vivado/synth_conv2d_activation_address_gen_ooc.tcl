set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

read_verilog -sv [file join $hw_dir "rtl" "lut_mul32x16_pipeline.sv"]
read_verilog -sv [file join $hw_dir "rtl" "conv2d_activation_address_gen.sv"]
synth_design -top conv2d_activation_address_gen -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]

report_utilization -file [file join $hw_dir "conv2d_activation_address_gen_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $hw_dir "conv2d_activation_address_gen_timing.rpt"]

puts "Conv2D activation address generator OOC synthesis completed."
