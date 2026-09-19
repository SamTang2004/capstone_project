# Resume a completed initial route without repeating synthesis and placement.
set root [file normalize [file join [file dirname [info script]] ..]]
set variant [lindex $argv 0]
if {$variant ni {banked pipelined}} { error "Expected banked or pipelined" }
set lanes 64
set pipeline [expr {$variant eq "pipelined"}]
set folder [file join $root build ${variant}_registered_impl scan${lanes}]
cd $folder
open_checkpoint routed.dcp
source [file join $root vivado finish_registered_impl.tcl]
