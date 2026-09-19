# Qualify all DUT paths between explicit on-chip launch and capture flops.
# Only fixture-port paths outside those flops are excluded, never DUT paths.
set root [file normalize [file join [file dirname [info script]] ..]]
set lanes 64
set pipeline 1
set merge 1
if {[llength $argv] > 0} { set lanes [lindex $argv 0] }
set variant pipelined
if {[llength $argv] > 1 && [lindex $argv 1] eq "banked"} {
    set pipeline 0
    set merge 0
    set variant banked
}
set folder [file join $root build ${variant}_registered_impl scan${lanes}]
file mkdir $folder
cd $folder
create_project -in_memory -part xc7z020clg400-1
foreach name {llm_work_finder8 llm_survivor_buffer8 llm_dot_product_lane llm_dot_core8 llm_pipelined_work_finder llm_wide_work_finder llm_wide_sparse_dot8 llm_sparse_dot_timing_boundary} {
    read_verilog -sv [file join $root rtl ${name}.sv]
}
set fd [open clock.xdc w]
puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
puts $fd {set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]}
puts $fd {set_false_path -from [get_ports fixture_inputs*]}
puts $fd {set_false_path -to [get_ports fixture_outputs*]}
close $fd
add_files -fileset constrs_1 clock.xdc
synth_design -top llm_sparse_dot_timing_boundary -generic [list SCAN_LANES=$lanes PIPELINE_MERGE=$pipeline MERGE_BANKS=$merge] -part xc7z020clg400-1 -mode out_of_context
if {[llength [get_cells -hier -filter {REF_NAME == DSP48E1}]] != 8} { error "Expected eight DSPs" }
if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black box" }
report_utilization -hierarchical -file utilization_synth.rpt
report_timing_summary -file timing_synth.rpt
write_checkpoint -force synth.dcp
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
source [file join $root vivado finish_registered_impl.tcl]
