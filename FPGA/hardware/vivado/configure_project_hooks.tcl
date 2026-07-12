set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
source [file join $script_dir "final_bitstream_sources.tcl"]
set default_project [file join $hw_dir "project_1" "project_1.xpr"]
set generated_project [file join $hw_dir "vivado_project" "budgetcuda_micro_gpu.xpr"]
set bitstream_hook [file join $script_dir "bitstream_standalone_drc_hook.tcl"]
set impl_run_tcl [file join $hw_dir "project_1" "project_1.runs" "impl_1" "${final_top}.tcl"]

if {[file exists $default_project]} {
    open_project $default_project
} elseif {[file exists $generated_project]} {
    open_project $generated_project
} else {
    source [file join $script_dir "create_project.tcl"]
}

set_property STEPS.WRITE_BITSTREAM.TCL.PRE $bitstream_hook [get_runs impl_1]

if {[llength [get_filesets -quiet sim_1]] != 0} {
    set_property top $final_sim_top [get_filesets sim_1]
}

update_compile_order -fileset sources_1
if {[llength [get_filesets -quiet sim_1]] != 0} {
    update_compile_order -fileset sim_1
}

puts "Configured write_bitstream pre-hook: $bitstream_hook"

if {[file exists $impl_run_tcl]} {
    set fp [open $impl_run_tcl r]
    set contents [read $fp]
    close $fp

    if {[string first "bitstream_standalone_drc_hook.tcl" $contents] < 0} {
        set marker "  write_bitstream -force ${final_top}.bit "
        set hook_line "  source [file nativename $bitstream_hook]\n"
        set contents [string map [list $marker "${hook_line}${marker}"] $contents]

        set fp [open $impl_run_tcl w]
        puts -nonewline $fp $contents
        close $fp
        puts "Patched generated implementation run Tcl: $impl_run_tcl"
    } else {
        puts "Generated implementation run Tcl already sources standalone DRC hook."
    }
}

close_project
