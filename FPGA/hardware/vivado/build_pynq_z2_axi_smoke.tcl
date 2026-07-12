# Builds a minimal board-level AXI-Lite diagnostic image at 0x43C80000.
set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set board_repo [file join $hw_dir "board_files" "XilinxBoardStore" "boards"]
set project_dir [file join $hw_dir "vivado_axi_smoke_project"]

if {![file exists [file join $board_repo "TUL" "pynq-z2" "A.0" "preset.xml"]]} {
    error "Missing vendored PYNQ-Z2 board files."
}

set_param board.repoPaths [list $board_repo]
create_project budgetcuda_axi_smoke $project_dir -part xc7z020clg400-1 -force
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

create_bd_design "axi_smoke"
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps7
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.0} \
] $ps7

set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]
set interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_gp0_interconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $interconnect
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_smoke_gpio]
set_property -dict [list CONFIG.C_ALL_INPUTS {1} CONFIG.C_GPIO_WIDTH {32}] $gpio

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_gp0_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_gp0_interconnect/M00_AXI] [get_bd_intf_pins axi_smoke_gpio/S_AXI]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins axi_gp0_interconnect/ACLK] \
    [get_bd_pins axi_gp0_interconnect/S00_ACLK] \
    [get_bd_pins axi_gp0_interconnect/M00_ACLK] \
    [get_bd_pins axi_smoke_gpio/s_axi_aclk] \
    [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins axi_gp0_interconnect/ARESETN] \
    [get_bd_pins axi_gp0_interconnect/S00_ARESETN] \
    [get_bd_pins axi_gp0_interconnect/M00_ARESETN] \
    [get_bd_pins axi_smoke_gpio/s_axi_aresetn]

set gpio_segment [lindex [get_bd_addr_segs -of_objects [get_bd_intf_pins axi_smoke_gpio/S_AXI]] 0]
assign_bd_address -offset 0x43C80000 -range 64K \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] $gpio_segment
validate_bd_design
save_bd_design
make_wrapper -files [get_files [file join $project_dir "budgetcuda_axi_smoke.srcs" "sources_1" "bd" "axi_smoke" "axi_smoke.bd"]] -top
add_files -norecurse [file join $project_dir "budgetcuda_axi_smoke.gen" "sources_1" "bd" "axi_smoke" "hdl" "axi_smoke_wrapper.v"]
set_property top axi_smoke_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
write_bitstream -force [file join $hw_dir "pynq_z2_axi_smoke.bit"]
puts "Built AXI-Lite smoke bitstream: [file join $hw_dir pynq_z2_axi_smoke.bit]"
