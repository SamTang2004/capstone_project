set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_dir [file join $hw_dir "vivado_project"]
source [file join $script_dir "final_bitstream_sources.tcl"]

create_project budgetcuda_micro_gpu $project_dir -part xc7z020clg400-1 -force

set_property target_language Verilog [current_project]

add_files -norecurse $final_rtl_files
add_files -fileset sim_1 -norecurse $final_sim_files
add_files -fileset constrs_1 -norecurse [file join $hw_dir "constraints.xdc"]

set_property file_type SystemVerilog [get_files *.sv]
set_property top $final_top [current_fileset]
set_property top $final_sim_top [get_filesets sim_1]
set_property STEPS.WRITE_BITSTREAM.TCL.PRE [file join $script_dir "bitstream_standalone_drc_hook.tcl"] [get_runs impl_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "Created Vivado project at $project_dir"
puts "Final PL top: $final_top"
