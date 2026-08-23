set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
source [file join $script_dir "final_bitstream_sources.tcl"]
read_verilog -sv $final_rtl_files
synth_design -top bcuda_transformer_ddr_top -part xc7z020clg400-1 -mode out_of_context
create_clock -name aclk -period 10.000 [get_ports aclk]
report_utilization -file [file join $hw_dir "reports" "transformer_top_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 -file [file join $hw_dir "reports" "transformer_top_timing.rpt"]
puts "Transformer DDR top OOC synthesis completed."
