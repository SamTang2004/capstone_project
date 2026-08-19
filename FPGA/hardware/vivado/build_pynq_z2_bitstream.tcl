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

launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
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
write_cfgmem -force -format bin -interface smapx32 -size 16 \
    -loadbit "up 0x0 [file join $hw_dir bcuda_pynq_z2.bit]" \
    -file [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force [file join $hw_dir "bcuda_pynq_z2.xsa"]
puts "Built PYNQ-Z2 bitstream: [file join $hw_dir bcuda_pynq_z2.bit]"
