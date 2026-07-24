set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
open_checkpoint [file join $hw_dir "vivado_board_project" "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper_routed.dcp"]
report_utilization -hierarchical -hierarchical_depth 8 -file [file join $hw_dir "pynq_z2_utilization_hier_impl.rpt"]
