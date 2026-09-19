set root [file normalize [file join [file dirname [info script]] ..]]
open_checkpoint [file join $root build sparse_board200 sparse_board.runs impl_1 sparse_ps_pl_wrapper_routed.dcp]
report_bus_skew -warn_on_violation -file [file join $root build sparse_release200 bus_skew.rpt]
