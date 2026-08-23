set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_dir [file join $hw_dir "vivado_board_project"]
set project_file [file join $project_dir "budgetcuda_board.xpr"]

open_project $project_file
open_run impl_1

set before_wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "Pre-optimization routed WNS: ${before_wns} ns"
if {$before_wns < 0.0} {
    phys_opt_design -directive AggressiveExplore
}

set run_dir [file join $project_dir "budgetcuda_board.runs" "impl_1"]
set routed_dcp [file join $run_dir "bcuda_ps_pl_wrapper_routed.dcp"]
set routed_bit [file join $run_dir "bcuda_ps_pl_wrapper.bit"]
write_checkpoint -force $routed_dcp
write_bitstream -force $routed_bit

report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -max_paths 50 -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
set setup_slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "Post-optimization timing: WNS=${setup_slack} ns WHS=${hold_slack} ns"
if {$setup_slack < 0.0 || $hold_slack < 0.0} {
    error "PYNQ-Z2 timing signoff failed: WNS=${setup_slack} ns WHS=${hold_slack} ns"
}

file copy -force $routed_bit [file join $hw_dir "bcuda_pynq_z2.bit"]
source [file join $script_dir "fpga_manager_bin.tcl"]
export_fpga_manager_bin $script_dir \
    [file join $hw_dir "bcuda_pynq_z2.bit"] [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force [file join $hw_dir "bcuda_pynq_z2.xsa"]

puts "Timing-closed PYNQ-Z2 release artifacts updated."
