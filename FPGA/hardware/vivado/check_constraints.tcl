set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set default_project [file join $hw_dir "project_1" "project_1.xpr"]
set generated_project [file join $hw_dir "vivado_project" "budgetcuda_micro_gpu.xpr"]

if {[file exists $default_project]} {
    open_project $default_project
} elseif {[file exists $generated_project]} {
    open_project $generated_project
} else {
    source [file join $script_dir "create_project.tcl"]
}

reset_run synth_1
launch_runs synth_1
wait_on_run synth_1
open_run synth_1

report_clocks
report_timing_summary -delay_type max -max_paths 10
