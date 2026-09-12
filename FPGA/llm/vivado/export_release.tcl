set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup_paths] || ![llength $hold_paths]} {error "Missing timing paths"}
set wns [get_property SLACK $setup_paths]
set whs [get_property SLACK $hold_paths]
if {$wns < 0 || $whs < 0} {error "Timing failed: WNS=$wns WHS=$whs; no release bitstream"}
# Register bitstream generation with the run manager so XSA can locate it.
close_design
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Bitstream run failed"}
set run_bit [file join [get_property DIRECTORY [get_runs impl_1]] bcuda_ps_pl_wrapper.bit]
if {![file exists $run_bit]} {error "Missing implementation bitstream: $run_bit"}
file copy -force $run_bit [file join $output_dir llm_pynq_z2.bit]
set hwh [file join $project_dir budgetcuda_board.gen sources_1 bd bcuda_ps_pl hw_handoff bcuda_ps_pl.hwh]
if {![file exists $hwh]} {error "Missing HWH: $hwh"}
file copy -force $hwh [file join $output_dir llm_pynq_z2.hwh]
write_hw_platform -fixed -include_bit -force [file join $output_dir llm_pynq_z2.xsa]
puts "LLM_RELEASE_PASS WNS=$wns WHS=$whs OUTPUT=$output_dir"
