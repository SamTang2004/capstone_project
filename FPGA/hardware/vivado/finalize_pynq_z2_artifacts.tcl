set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set project_dir [file join $hw_dir "vivado_board_project"]

open_project [file join $project_dir "budgetcuda_board.xpr"]
open_run impl_1
report_utilization -file [file join $hw_dir "pynq_z2_utilization_impl.rpt"]
report_timing_summary -file [file join $hw_dir "pynq_z2_timing_impl.rpt"]
write_bitstream -force [file join $hw_dir "bcuda_pynq_z2.bit"]
write_cfgmem -force -format bin -interface smapx32 -size 16 \
    -loadbit "up 0x0 [file join $hw_dir bcuda_pynq_z2.bit]" \
    -file [file join $hw_dir "bcuda_pynq_z2.bin"]
write_hw_platform -fixed -include_bit -force \
    [file join $hw_dir "bcuda_pynq_z2.xsa"]
puts "Finalized PYNQ-Z2 artifacts from routed implementation."
