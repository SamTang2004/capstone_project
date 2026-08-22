set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set checkpoint [file join $hw_dir "vivado_board_project" "budgetcuda_board.runs" \
    "impl_1" "bcuda_ps_pl_wrapper_routed.dcp"]

if {![file exists $checkpoint]} {
    error "Routed checkpoint not found: $checkpoint"
}

open_checkpoint $checkpoint

set detailed_report [file join $hw_dir "pynq_z2_timing_hotspots.rpt"]
set summary_report [file join $hw_dir "pynq_z2_timing_hotspots.tsv"]
set paths [get_timing_paths -delay_type max -max_paths 100 -nworst 1]

report_timing -delay_type max -max_paths 100 -nworst 1 \
    -path_type full_clock_expanded -input_pins -file $detailed_report

set output [open $summary_report w]
puts $output "rank\tslack_ns\trequirement_ns\tdatapath_ns\tlogic_levels\tstartpoint\tendpoint"
set rank 0
foreach path $paths {
    incr rank
    set startpoint [get_property STARTPOINT_PIN $path]
    set endpoint [get_property ENDPOINT_PIN $path]
    puts $output [join [list \
        $rank \
        [get_property SLACK $path] \
        [get_property REQUIREMENT $path] \
        [get_property DATAPATH_DELAY $path] \
        [get_property LOGIC_LEVELS $path] \
        [get_property NAME $startpoint] \
        [get_property NAME $endpoint]] "\t"]
}
close $output

set worst [lindex $paths 0]
puts "TIMING_HOTSPOT_REPORT=$detailed_report"
puts "TIMING_HOTSPOT_TABLE=$summary_report"
puts "WORST_SLACK=[get_property SLACK $worst]"
puts "WORST_START=[get_property NAME [get_property STARTPOINT_PIN $worst]]"
puts "WORST_END=[get_property NAME [get_property ENDPOINT_PIN $worst]]"
puts "WORST_DATAPATH=[get_property DATAPATH_DELAY $worst]"
puts "WORST_LEVELS=[get_property LOGIC_LEVELS $worst]"
