set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_file [file join $hw_dir "vivado_board_project" "budgetcuda_board.xpr"]

open_project $project_file
open_run impl_1
phys_opt_design -directive AggressiveExplore
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
write_checkpoint -force [file join $hw_dir "vivado_board_project" "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper_physopt.dcp"]
write_bitstream -force [file join $hw_dir "bcuda_pynq_z2.bit"]

set worst_path [get_timing_paths -delay_type max -max_paths 1]
puts "Post-physopt WNS: [get_property SLACK $worst_path] ns"
close_project
