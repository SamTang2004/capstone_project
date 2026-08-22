set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set output_dir [expr {[info exists ::env(BCUDA_STRESS_OUTPUT)]
    ? [file normalize $::env(BCUDA_STRESS_OUTPUT)]
    : [file join $hw_dir "xsim_memory_io_stress"]}]
file mkdir $output_dir
cd $output_dir

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]
set rtl [file join $hw_dir "rtl" "memory_io_driver.sv"]
set tb [file join $hw_dir "sim" "tb_memory_io_load_balance_stress.sv"]
set stress_top [expr {[info exists ::env(BCUDA_STRESS_TOP)]
    ? $::env(BCUDA_STRESS_TOP)
    : "tb_memory_io_load_balance_stress"}]
set sim_name "${stress_top}_sim"

if {[catch {exec $xvlog "--sv" $rtl $tb} output options]} {
    puts $output
    error "xvlog failed: $options"
}
puts $output
if {[catch {exec $xelab $stress_top "-L" xpm "--timescale" 1ns/1ps "-s" $sim_name} output options]} {
    puts $output
    error "xelab failed: $options"
}
puts $output
if {[catch {exec $xsim $sim_name "-runall"} output options]} {
    puts $output
    error "xsim failed: $options"
}
puts $output
if {![regexp {load-balance stress golden trace passed} $output]} {
    error "stress simulation did not report a golden trace pass"
}
