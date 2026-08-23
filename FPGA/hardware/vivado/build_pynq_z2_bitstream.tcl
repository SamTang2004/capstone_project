set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
# The board wrapper includes PS clocking, address assignment, and HP0 wiring.
# Regenerate it for every build so a Tcl change cannot leave a stale block
# design behind the newly generated bitstream.
source [file join $script_dir "create_ps_pl_wrapper.tcl"]

set build_jobs 8
if {[info exists ::env(BCUDA_BUILD_JOBS)]} {
    set build_jobs $::env(BCUDA_BUILD_JOBS)
}

# A stopped or failed run cannot be relaunched. Always rebuild the board
# artifact from the current RTL and block design sources.
if {[llength [get_runs -quiet impl_1]]} {
    reset_run impl_1
}
foreach run [get_runs -quiet *synth_1] {
    reset_run $run
}

launch_runs synth_1 -jobs $build_jobs
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $hw_dir "pynq_z2_utilization_synth.rpt"]

# The design is dense enough that the default router can lose timing after a
# clean physical-optimization result. Use its timing-focused search directly
# instead of relying on a memory-heavy post-route recovery pass.
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
wait_on_run impl_1
open_run impl_1

# Dense Zynq-7020 placements can lose several tenths of a nanosecond during
# routing even when placement timing is clean. Recover those local route
# delays before release, then reject any image that still violates setup or
# hold timing instead of silently exporting an overclocked bitstream.
set setup_path [get_timing_paths -delay_type max -max_paths 1]
set setup_slack [get_property SLACK $setup_path]
if {$setup_slack < 0.0} {
    puts "Post-route WNS is ${setup_slack} ns; running AggressiveExplore"
    phys_opt_design -directive AggressiveExplore
    set routed_dcp [file join $project_dir "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper_routed.dcp"]
    write_checkpoint -force $routed_dcp
    write_bitstream -force [file join $project_dir "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper.bit"]
}

report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
set setup_slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup_slack < 0.0 || $hold_slack < 0.0} {
    error "PYNQ-Z2 timing signoff failed: WNS=${setup_slack} ns WHS=${hold_slack} ns"
}
puts "PYNQ-Z2 timing signoff passed: WNS=${setup_slack} ns WHS=${hold_slack} ns"
set generated_bits [glob -nocomplain [file join $project_dir "budgetcuda_board.runs" "impl_1" "*.bit"]]
if {[llength $generated_bits] != 1} {
    error "Expected one implementation bitstream, found [llength $generated_bits]: $generated_bits"
}
file copy -force [lindex $generated_bits 0] [file join $hw_dir "bcuda_pynq_z2.bit"]
set generated_hwh [file join $project_dir "budgetcuda_board.gen" "sources_1" "bd" "bcuda_ps_pl" "hw_handoff" "bcuda_ps_pl.hwh"]
if {![file exists $generated_hwh]} {
    error "Vivado did not generate the production HWH at $generated_hwh"
}
file copy -force $generated_hwh [file join $hw_dir "bcuda_pynq_z2.hwh"]
# fpga_manager consumes a raw byte-swapped configuration payload, not the
# SMAPx32 flash image produced by write_cfgmem.
source [file join $script_dir "fpga_manager_bin.tcl"]
export_fpga_manager_bin $script_dir \
    [file join $hw_dir "bcuda_pynq_z2.bit"] [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force [file join $hw_dir "bcuda_pynq_z2.xsa"]
puts "Built PYNQ-Z2 bitstream: [file join $hw_dir bcuda_pynq_z2.bit]"
