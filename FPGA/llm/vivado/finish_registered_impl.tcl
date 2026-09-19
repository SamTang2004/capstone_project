# Shared finalization, used after initial routing or checkpoint resume.
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup < 0 || $hold < 0} {
    report_timing_summary -file timing_initial_route.rpt
    write_checkpoint -force initial_route.dcp
    if {$hold < 0} {
        phys_opt_design -directive ExploreWithAggressiveHoldFix
    } else {
        phys_opt_design -directive AggressiveExplore
    }
    route_design -directive Explore
}
report_utilization -hierarchical -file utilization_routed.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file timing_routed.rpt
report_timing -max_paths 20 -file critical_setup.rpt
report_timing -delay_type min -max_paths 20 -file critical_hold.rpt
report_route_status -file route_status.rpt
report_drc -file drc.rpt
report_exceptions -file exceptions.rpt
write_checkpoint -force routed.dcp
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "RESULT: registered boundary lanes=$lanes pipeline=$pipeline routed setup=$setup hold=$hold period=10ns"
if {$setup < 0 || $hold < 0} { error "Routed timing target not met" }
puts "PASS: registered boundary routed 100MHz lanes=$lanes pipeline=$pipeline"
