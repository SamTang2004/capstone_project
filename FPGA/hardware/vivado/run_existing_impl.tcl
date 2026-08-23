# Complete implementation from an already synthesized board project.
# This keeps a long routed run independent of the shell command timeout.
set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_dir [file join $hw_dir "vivado_board_project"]

open_project [file join $project_dir "budgetcuda_board.xpr"]
if {[llength [get_runs -quiet impl_1]]} {
    reset_run impl_1
}
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
write_bitstream -force [file join $hw_dir "bcuda_pynq_z2.bit"]
source [file join $script_dir "fpga_manager_bin.tcl"]
export_fpga_manager_bin $script_dir \
    [file join $hw_dir "bcuda_pynq_z2.bit"] [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force [file join $hw_dir "bcuda_pynq_z2.xsa"]
puts "Implemented existing PYNQ-Z2 project."
