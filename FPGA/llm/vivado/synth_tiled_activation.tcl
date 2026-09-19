set root [file normalize [file join [file dirname [info script]] ..]]
foreach top {llm_tiled_activation8 llm_activation_block_map} {
    set folder [file join $root build tiled_activation_synth $top]
    file mkdir $folder
    cd $folder
    create_project -in_memory -part xc7z020clg400-1
    foreach name {llm_bitmap_finder64 llm_dot_product_lane llm_dot_core8 llm_activation_block_map llm_tiled_activation8} {
        read_verilog -sv [file join $root rtl ${name}.sv]
    }
    set fd [open clock.xdc w]
    puts $fd {create_clock -name clk -period 10.000 [get_ports clk]}
    puts $fd {set_clock_uncertainty 0.300 [get_clocks clk]}
    close $fd
    add_files -fileset constrs_1 clock.xdc
    synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
    set expected [expr {$top eq "llm_tiled_activation8" ? 8 : 0}]
    if {[llength [get_cells -hier -quiet -filter {REF_NAME == DSP48E1}]] != $expected} {error "Unexpected DSP count"}
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] != 0} {error "Black boxes"}
    report_utilization -file utilization.rpt
    report_utilization -hierarchical -file utilization_hier.rpt
    report_timing_summary -file timing_synth.rpt
    write_checkpoint -force synth.dcp
    puts "PASS: tiled synthesis top=$top"
    close_project
}
