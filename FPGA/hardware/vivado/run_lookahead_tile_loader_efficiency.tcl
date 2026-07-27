set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set sim_work [file join $hw_dir "xsim_lookahead_tile_loader_efficiency"]
file mkdir $sim_work
cd $sim_work

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]
set top tb_lookahead_tile_loader_efficiency
set snapshot "${top}_sim"

set sources [list \
    [file join $hw_dir "rtl" "lookahead_tile_loader_int4.sv"] \
    [file join $hw_dir "sim" "tb_lookahead_tile_loader_efficiency.sv"] \
]
set compile_cmd [list $xvlog "--sv"]
foreach source $sources {
    lappend compile_cmd [file nativename $source]
}

if {[catch {exec {*}$compile_cmd} output options]} {
    puts $output
    error "TileLoader efficiency xvlog failed: $options"
}
puts $output
if {[catch {exec $xelab $top "-s" $snapshot "-timescale" "1ns/1ps"} output options]} {
    puts $output
    error "TileLoader efficiency elaboration failed: $options"
}
puts $output
if {[catch {exec $xsim $snapshot "-runall"} output options]} {
    puts $output
    error "TileLoader efficiency simulation failed: $options"
}
puts $output
if {![regexp {TileLoader efficiency benchmark passed} $output]} {
    error "TileLoader efficiency benchmark did not report a pass"
}
