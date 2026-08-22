set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]

source [file join $script_dir "final_bitstream_sources.tcl"]

foreach rtl_file $final_rtl_files {
    read_verilog -sv $rtl_file
}

synth_design -top $final_top -part xc7z020clg400-1 -mode out_of_context
create_clock -name aclk -period 10.000 [get_ports aclk]

report_utilization -hierarchical \
    -file [file join $hw_dir "final_top_wide_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $hw_dir "final_top_wide_timing.rpt"]

puts "Canonical wide-read production top OOC synthesis completed."
