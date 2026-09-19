# Routed shared-clock boundary qualification for bitmap streaming only.
set root [file normalize [file join [file dirname [info script]] ..]]
set folder [file join $root build activation_sharing_registered_impl]
file mkdir $folder
cd $folder
create_project -in_memory -part xc7z020clg400-1
foreach name {llm_work_finder8 llm_pipelined_work_finder llm_bitmap_finder64 llm_dot_product_lane llm_dot_core8 llm_shared_activation8 llm_shared_activation_timing_boundary} {
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
synth_design -top llm_shared_activation_timing_boundary -part xc7z020clg400-1 -mode out_of_context
if {[llength [get_cells -hier -filter {REF_NAME == DSP48E1}]] != 8} { error "Expected eight DSPs" }
if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black boxes" }
report_utilization -hierarchical -file utilization_synth.rpt
write_checkpoint -force synth.dcp
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup < 0 || $hold < 0} {
    report_timing_summary -file timing_initial_route.rpt
    write_checkpoint -force initial_route.dcp
    if {$hold < 0} { phys_opt_design -directive ExploreWithAggressiveHoldFix } else { phys_opt_design -directive AggressiveExplore }
    route_design -directive Explore
}
report_utilization -hierarchical -file utilization_routed.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file timing_routed.rpt
report_route_status -file route_status.rpt
report_drc -file drc.rpt
report_exceptions -file exceptions.rpt
write_checkpoint -force routed.dcp
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "RESULT: shared activation streaming routed setup=$setup hold=$hold period=10ns"
if {$setup < 0 || $hold < 0} { error "Routed 100MHz target not met" }
puts "PASS: shared activation streaming registered boundary 100MHz"
