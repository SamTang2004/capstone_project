set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set sim_work [file join $hw_dir "xsim_lookahead_tile_loader"]
file mkdir $sim_work
cd $sim_work

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]
set top tb_lookahead_tile_loader_int4
set snapshot "${top}_sim"

set sources [list \
    [file join $hw_dir "rtl" "lookahead_tile_loader_int4.sv"] \
    [file join $hw_dir "sim" "tb_lookahead_tile_loader_int4.sv"] \
]
set compile_cmd [list $xvlog "--sv"]
foreach source $sources {
    lappend compile_cmd [file nativename $source]
}

puts "===== Compiling look-ahead INT4 TileLoader ====="
if {[catch {exec {*}$compile_cmd} output options]} {
    puts $output
    error "xvlog failed: $options"
}
puts $output

puts "===== Elaborating $top ====="
if {[catch {exec $xelab $top "-s" $snapshot "-timescale" "1ns/1ps"} output options]} {
    puts $output
    error "xelab failed: $options"
}
puts $output

puts "===== Running $top ====="
if {[catch {exec $xsim $snapshot "-runall"} output options]} {
    puts $output
    error "xsim failed: $options"
}
puts $output
if {![regexp {look-ahead TileLoader golden trace passed} $output]} {
    error "$top did not report a golden trace pass"
}
