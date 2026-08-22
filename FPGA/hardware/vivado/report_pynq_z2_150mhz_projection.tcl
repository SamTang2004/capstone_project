set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_dir [file join $hw_dir "vivado_board_project"]
set project_file [file join $project_dir "budgetcuda_board.xpr"]

if {![file exists $project_file]} {
    error "Vivado project not found at $project_file"
}

open_project $project_file
open_run impl_1

set clock_source [get_pins bcuda_ps_pl_i/processing_system7_0/inst/PS7_i/FCLKCLK[0]]
if {[llength $clock_source] != 1} {
    error "Expected one PS FCLK0 source pin, found [llength $clock_source]"
}

# This is a report-only constraint override. The routed checkpoint and release
# artifacts remain unchanged.
create_clock -period 6.667 -name clk_fpga_0 $clock_source
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $hw_dir "pynq_z2_timing_150mhz_projection.rpt"]

set setup_slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "PYNQ-Z2 150 MHz projection: WNS=${setup_slack} ns WHS=${hold_slack} ns"
close_project
