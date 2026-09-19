# Routed OOC block experiment. IO budgets describe a synchronous local interface,
# not board pins or a DDR controller. No production project or bitstream changes.
set root [file normalize [file join [file dirname [info script]] ..]]
set lanes 64
if {[llength $argv] > 0} { set lanes [lindex $argv 0] }
set folder [file join $root build pipelined_sparse_dot_impl scan${lanes}]
set pipeline 1
set merge 1
if {[llength $argv] > 1 && [lindex $argv 1] eq "banked"} {
    set pipeline 0
    set merge 0
    set folder [file join $root build banked_sparse_dot_impl scan${lanes}]
}
file mkdir $folder
cd $folder
create_project -in_memory -part xc7z020clg400-1
foreach name {llm_work_finder8 llm_survivor_buffer8 llm_dot_product_lane llm_dot_core8 llm_pipelined_work_finder llm_wide_work_finder llm_wide_sparse_dot8} {
    read_verilog -sv [file join $root rtl ${name}.sv]
}
set fd [open clock.xdc w]
puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
puts $fd {set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]}
puts $fd {set_input_delay -clock clk -max 2.000 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
puts $fd {set_input_delay -clock clk -min 0.500 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
puts $fd {set_output_delay -clock clk -max 2.000 [all_outputs]}
puts $fd {set_output_delay -clock clk -min 0.500 [all_outputs]}
close $fd
add_files -fileset constrs_1 clock.xdc
synth_design -top llm_wide_sparse_dot8 -generic [list SCAN_LANES=$lanes MERGE_BANKS=$merge PIPELINE_MERGE=$pipeline] -part xc7z020clg400-1 -mode out_of_context
if {[llength [get_cells -hier -filter {REF_NAME == DSP48E1}]] != 8} { error "Expected eight DSPs" }
if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black box" }
report_utilization -file utilization_synth.rpt
report_timing_summary -file timing_synth.rpt
write_checkpoint -force synth.dcp
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
report_utilization -file utilization_routed.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file timing_routed.rpt
report_route_status -file route_status.rpt
report_drc -file drc.rpt
write_checkpoint -force routed.dcp
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "RESULT: sparse dot lanes=$lanes pipeline=$pipeline routed setup=$setup hold=$hold period=10ns"
if {$setup < 0 || $hold < 0} { error "Routed timing target not met" }
puts "PASS: sparse dot routed 100MHz lanes=$lanes pipeline=$pipeline"
