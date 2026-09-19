# Isolated area/timing comparison, without changing the production board project.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build wide_sparse_dot_synth]
set merge 0
if {[llength $argv] > 0 && [lindex $argv 0] eq "merge"} {
    set merge 1
    set out [file join $root build merged_sparse_dot_synth]
}
file mkdir $out
set sources {}
foreach name {llm_work_finder8 llm_survivor_buffer8 llm_dot_product_lane llm_dot_core8 llm_pipelined_work_finder llm_wide_work_finder llm_wide_sparse_dot8} {
    lappend sources [file join $root rtl ${name}.sv]
}
foreach lanes {8 16 32 64} {
    set folder [file join $out scan${lanes}]
    file mkdir $folder
    cd $folder
    create_project -in_memory -part xc7z020clg400-1
    read_verilog -sv $sources
    set fd [open clock.xdc w]
    puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
    puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
    close $fd
    add_files -fileset constrs_1 clock.xdc
    synth_design -top llm_wide_sparse_dot8 -generic [list SCAN_LANES=$lanes MERGE_BANKS=$merge] -part xc7z020clg400-1 -mode out_of_context
    set dsps [get_cells -hier -filter {REF_NAME == DSP48E1}]
    if {[llength $dsps] != 8} { error "Expected eight DSPs, found [llength $dsps]" }
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black box" }
    report_utilization -file utilization.rpt
    report_timing_summary -file timing_synth.rpt
    write_checkpoint -force wide_sparse_dot_synth.dcp
    puts "PASS: wide sparse dot synthesis lanes=$lanes DSPs=8"
    close_project
}
