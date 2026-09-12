set entry_dir [file dirname [file normalize [info script]]]
set llm_dir [file normalize [file join $entry_dir ..]]
set project_dir [file join $llm_dir build gemm_board]
set output_dir [file join $llm_dir build gemm_release]
set jobs 4
open_project [file join $project_dir budgetcuda_board.xpr]
open_run impl_1
source [file join $entry_dir export_release.tcl]
