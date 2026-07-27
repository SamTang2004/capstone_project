set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "dual_int4_ddr_top_ooc"]
file mkdir $report_dir

source [file join $script_dir "final_bitstream_sources.tcl"]
foreach source $final_rtl_files {
    read_verilog -sv $source
}

synth_design \
    -top bcuda_dual_int4_ddr_top \
    -part xc7z020clg400-1 \
    -mode out_of_context \
    -flatten_hierarchy rebuilt

create_clock -name int4_top_clk -period 10.000 [get_ports aclk]

report_utilization \
    -hierarchical \
    -file [file join $report_dir "utilization.rpt"]
report_timing_summary \
    -delay_type max \
    -max_paths 20 \
    -file [file join $report_dir "timing.rpt"]

puts "Dual INT4 DDR top utilization: [file join $report_dir utilization.rpt]"
puts "Dual INT4 DDR top timing: [file join $report_dir timing.rpt]"
