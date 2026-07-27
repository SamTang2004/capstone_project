set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set sim_work [file join $hw_dir "xsim_int4_streaming_rows"]
file mkdir $sim_work
cd $sim_work

set vivado_bin [file dirname [info nameofexecutable]]
if {![file exists [file join $vivado_bin "xvlog.bat"]]} {
    set vivado_bin [file normalize [file join $vivado_bin ".." ".."]]
}
set xvlog [file join $vivado_bin "xvlog.bat"]
set xelab [file join $vivado_bin "xelab.bat"]
set xsim [file join $vivado_bin "xsim.bat"]

set sources [list \
    [file join $hw_dir "rtl" "int4_streaming_scalar_row_packer.sv"] \
    [file join $hw_dir "rtl" "int4_dual_array_row_stream_buffer.sv"] \
    [file join $hw_dir "sim" "tb_int4_streaming_scalar_row_packer.sv"] \
    [file join $hw_dir "sim" "tb_int4_dual_array_row_stream_buffer.sv"] \
]
set compile_cmd [list $xvlog "--sv"]
foreach source $sources {
    lappend compile_cmd [file nativename $source]
}
if {[catch {exec {*}$compile_cmd} output options]} {
    puts $output
    error "Streaming-row xvlog failed: $options"
}
puts $output

foreach top {
    tb_int4_streaming_scalar_row_packer
    tb_int4_dual_array_row_stream_buffer
} {
    set snapshot "${top}_sim"
    if {[catch {exec $xelab $top "-s" $snapshot "-timescale" "1ns/1ps"} output options]} {
        puts $output
        error "$top elaboration failed: $options"
    }
    puts $output
    if {[catch {exec $xsim $snapshot "-runall"} output options]} {
        puts $output
        error "$top simulation failed: $options"
    }
    puts $output
    if {![regexp {passed} $output]} {
        error "$top did not report a pass"
    }
}

puts "INT4 streaming-row regression passed."
