set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
source [file join $script_dir "final_bitstream_sources.tcl"]

set sim_work [file join $hw_dir "xsim_final_top"]
file mkdir $sim_work
cd $sim_work

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]

set compile_cmd [list $xvlog "--sv"]
foreach f [concat $final_rtl_files $final_sim_files] {
    lappend compile_cmd [file nativename $f]
}

puts "===== Compiling final Budget-CUDA PL top and dependencies ====="
if {[catch {exec {*}$compile_cmd} compile_output compile_options]} {
    puts $compile_output
    error "xvlog failed: $compile_options"
}
puts $compile_output

set sim_name "${final_sim_top}_sim"
puts "===== Elaborating $final_sim_top ====="
if {[catch {exec $xelab $final_sim_top "-s" $sim_name "-timescale" "1ns/1ps"} elab_output elab_options]} {
    puts $elab_output
    error "xelab failed for $final_sim_top: $elab_options"
}
puts $elab_output

puts "===== Running $final_sim_top ====="
if {[catch {exec $xsim $sim_name "-runall"} sim_output sim_options]} {
    puts $sim_output
    error "xsim failed for $final_sim_top: $sim_options"
}
puts $sim_output

if {![regexp {golden trace passed} $sim_output]} {
    error "$final_sim_top did not report a golden trace pass"
}

puts "Final Budget-CUDA PL top simulation completed."
