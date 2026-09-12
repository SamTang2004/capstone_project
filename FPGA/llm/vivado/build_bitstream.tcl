set entry_dir [file dirname [file normalize [info script]]]
source [file join $entry_dir create_board.tcl]
set output_dir [file join $llm_dir build gemm_release]
file mkdir $output_dir
set jobs 4
if {[info exists ::env(LLM_BUILD_JOBS)]} {set jobs $::env(LLM_BUILD_JOBS)}
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
open_run synth_1
report_utilization -file [file join $output_dir utilization_synth.rpt]
report_utilization -hierarchical -file [file join $output_dir hierarchy_synth.rpt]
set forbidden [get_cells -hier -quiet -filter {REF_NAME =~ *rowwise* || REF_NAME =~ *heatvit* || REF_NAME =~ *requantize* || REF_NAME =~ *elementwise* || REF_NAME =~ *layernorm* || REF_NAME =~ *softmax* || REF_NAME =~ *gelu*}]
if {[llength $forbidden]} {error "Unexpected DeiT operators remain: $forbidden"}
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -file [file join $output_dir utilization_impl.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output_dir timing_impl.rpt]
report_drc -file [file join $output_dir drc.rpt]
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup_paths] || ![llength $hold_paths]} {error "Missing timing paths"}
set wns [get_property SLACK $setup_paths]
set whs [get_property SLACK $hold_paths]
if {$wns < 0 || $whs < 0} {error "Timing failed: WNS=$wns WHS=$whs; no release bitstream"}
write_bitstream -force [file join $output_dir llm_pynq_z2.bit]
set hwh [file join $project_dir budgetcuda_board.gen sources_1 bd bcuda_ps_pl hw_handoff bcuda_ps_pl.hwh]
if {![file exists $hwh]} {error "Missing HWH: $hwh"}
file copy -force $hwh [file join $output_dir llm_pynq_z2.hwh]
write_hw_platform -fixed -include_bit -force [file join $output_dir llm_pynq_z2.xsa]
puts "LLM_RELEASE_PASS WNS=$wns WHS=$whs FCLK_MHZ=$pl_clock_mhz OUTPUT=$output_dir"
