set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set checkpoint [file join $hw_dir "vivado_board_project" "budgetcuda_board.runs" "impl_1" "bcuda_ps_pl_wrapper_routed.dcp"]

open_checkpoint $checkpoint
write_verilog -force -mode funcsim [file join $hw_dir "routed_hp0_funcsim.v"]

foreach pattern [list "*ddr_bridge*len_r_reg*" "*ddr_bridge*m_axi_arlen*" "*processing_system7_0*S_AXI_HP0_ARLEN*"] {
    puts "===== $pattern ====="
    foreach object [concat [get_cells -quiet -hier $pattern] [get_nets -quiet -hier $pattern] [get_pins -quiet -hier $pattern]] {
        puts $object
    }
}
