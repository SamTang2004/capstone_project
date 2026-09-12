# Optional recovery of this project's routed netlist, without changing RTL/clock.
set llm_dir [file normalize [file join [file dirname [info script]] ..]]
set board_dir [file join $llm_dir build board]
set out [file join $llm_dir build postroute]
file mkdir $out
open_checkpoint [file join $board_dir budgetcuda_board.runs impl_1 bcuda_ps_pl_wrapper_routed.dcp]
phys_opt_design -directive AggressiveExplore
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
report_drc -file [file join $out drc.rpt]
write_checkpoint -force [file join $out llm_optimized.dcp]
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$wns < 0 || $whs < 0} {error "Post-route timing failed: WNS=$wns WHS=$whs"}
write_bitstream -force [file join $out llm_pynq_z2.bit]
file copy -force [file join $board_dir budgetcuda_board.gen sources_1 bd bcuda_ps_pl hw_handoff bcuda_ps_pl.hwh] [file join $out llm_pynq_z2.hwh]
puts "LLM_POSTROUTE_PASS WNS=$wns WHS=$whs OUTPUT=$out"
