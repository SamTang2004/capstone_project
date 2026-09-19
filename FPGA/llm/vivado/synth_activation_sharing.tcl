# Exploratory synthesis only. A clock constraint is not routed timing closure.
set root [file normalize [file join [file dirname [info script]] ..]]
set sources {}
foreach name {llm_work_finder8 llm_pipelined_work_finder llm_bitmap_finder64 llm_dot_product_lane llm_dot_core8 llm_shared_activation8} {
    lappend sources [file join $root rtl ${name}.sv]
}
foreach mode {0 1 2} {
    set folder [file join $root build activation_sharing_synth mode${mode}]
    file mkdir $folder
    cd $folder
    create_project -in_memory -part xc7z020clg400-1
    read_verilog -sv $sources
    set fd [open clock.xdc w]
    puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
    puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
    close $fd
    add_files -fileset constrs_1 clock.xdc
    set bitmap [expr {$mode != 0}]
    set streaming [expr {$mode == 2}]
    synth_design -top llm_shared_activation8 -generic [list BITMAP=$bitmap STREAMING=$streaming] -part xc7z020clg400-1 -mode out_of_context
    if {[llength [get_cells -hier -filter {REF_NAME == DSP48E1}]] != 8} { error "Expected eight DSPs" }
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} { error "Unexpected black boxes" }
    report_utilization -file utilization.rpt
    report_utilization -hierarchical -file utilization_hier.rpt
    report_timing_summary -file timing_synth.rpt
    write_checkpoint -force activation_sharing.dcp
    puts "PASS: activation sharing synthesis MODE=$mode"
    close_project
}
