set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
source [file join $script_dir "final_bitstream_sources.tcl"]
set default_project [file join $hw_dir "project_1" "project_1.xpr"]
set generated_project [file join $hw_dir "vivado_project" "budgetcuda_micro_gpu.xpr"]
set bit_file [file join $hw_dir $final_bitstream_name]

if {[file exists $default_project]} {
    open_project $default_project
} elseif {[file exists $generated_project]} {
    open_project $generated_project
} else {
    source [file join $script_dir "create_project.tcl"]
}

if {[catch {open_run impl_1} open_run_error]} {
    error "Implementation run impl_1 is not openable. Run implementation first. Detail: $open_run_error"
}

# The standalone RTL top intentionally exposes integration ports before the
# PS/PL block design exists. They are not real package pins in the final system.
source [file join $script_dir "bitstream_standalone_drc_hook.tcl"]

write_bitstream -force $bit_file
puts "Wrote standalone development bitstream for $final_top: $bit_file"
