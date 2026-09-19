# Compare isolated implementations; never edits or programs the board project.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build sparse_dot_synth]
file mkdir $out
set sources {}
foreach name {llm_work_finder8 llm_survivor_buffer8 llm_work_packer8 llm_dot_product_lane llm_dot_core8 llm_sparse_dot8} {
    lappend sources [file join $root rtl ${name}.sv]
}
foreach mode {0 1 2} {
    set mode_out [file join $out mode${mode}]
    file mkdir $mode_out
    cd $mode_out
    create_project -in_memory -part xc7z020clg400-1
    read_verilog -sv $sources
    set fd [open clock.xdc w]
    puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
    puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
    close $fd
    add_files -fileset constrs_1 clock.xdc
    synth_design -top llm_sparse_dot8 -generic MODE=$mode -part xc7z020clg400-1 -mode out_of_context
    set dsps [get_cells -hier -filter {REF_NAME == DSP48E1}]
    if {[llength $dsps] != 8} { error "Expected eight DSPs, found [llength $dsps]" }
    if {[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black box" }
    report_utilization -file utilization.rpt
    report_timing_summary -file timing_synth.rpt
    write_checkpoint -force sparse_dot_synth.dcp
    puts "PASS: sparse dot synthesis mode=$mode DSPs=8"
    close_project
}
