set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
# The board wrapper includes PS clocking, address assignment, and HP0 wiring.
# Regenerate it for every build so a Tcl change cannot leave a stale block
# design behind the newly generated bitstream.
source [file join $script_dir "create_ps_pl_wrapper.tcl"]

# A stopped or failed run cannot be relaunched. Always rebuild the board
# artifact from the current RTL and block design sources.
if {[llength [get_runs -quiet impl_1]]} {
    reset_run impl_1
}
foreach run [get_runs -quiet *synth_1] {
    reset_run $run
}

launch_runs synth_1 -jobs 8
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $hw_dir "pynq_z2_utilization_synth.rpt"]

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
set routed_wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
if {$routed_wns < 0.0} {
    puts "Routed WNS is $routed_wns ns; running post-route physical optimization."
    phys_opt_design -directive AggressiveExplore
    write_checkpoint -force [file join $project_dir "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper_physopt.dcp"]
}
report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
write_bitstream -force [file join $hw_dir "bcuda_pynq_z2.bit"]
write_cfgmem -force -format bin -interface smapx32 -size 16 \
    -loadbit "up 0x0 [file join $hw_dir bcuda_pynq_z2.bit]" \
    -file [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force [file join $hw_dir "bcuda_pynq_z2.xsa"]
puts "Built PYNQ-Z2 bitstream: [file join $hw_dir bcuda_pynq_z2.bit]"
