# Creates the board-facing PS/PL control wrapper around the Budget-CUDA RTL.
#
# The PYNQ-Z2 board files are vendored under hardware/board_files so the PS7
# DDR/MIO preset is reproducible and does not depend on a global Vivado install.

set llm_dir [file normalize [file join [file dirname [info script]] ".."]]
set script_dir [file normalize [file join $llm_dir ".." "hardware" "vivado"]]
set hw_dir [file normalize [file join $script_dir ".."]]
set board_repo [file join $hw_dir "board_files" "XilinxBoardStore" "boards"]
if {![file exists [file join $board_repo "TUL" "pynq-z2" "A.0" "preset.xml"]]} {
    error "Missing vendored PYNQ-Z2 board files. Restore hardware/board_files/XilinxBoardStore."
}

set manifest [open [file join $llm_dir vivado rtl_sources.txt] r]
set llm_rtl_files {}
foreach relative_path [split [read $manifest] "\n"] {
    if {[string trim $relative_path] ne ""} {
        lappend llm_rtl_files [file normalize [file join $llm_dir vivado [string trim $relative_path]]]
    }
}
close $manifest
set project_dir [file join $llm_dir "build" "gemm_board"]
set_param board.repoPaths [list $board_repo]
create_project budgetcuda_board $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
add_files -norecurse $llm_rtl_files
add_files -norecurse [list [file join $hw_dir rtl memory_io_axi3_bridge.sv] [file join $hw_dir rtl memory_io_axi3_bridge_ip.v] [file join $llm_dir rtl llm_axi_queued_ip.v]]
add_files -fileset constrs_1 -norecurse [file join $script_dir "pynq_z2_timing_margin.xdc"]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1

create_bd_design "bcuda_ps_pl"
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps7

# Keep 100 MHz as the production default, but permit a lower bring-up clock
# without maintaining a second block-design script.  The chosen frequency is
# recorded in the generated HWH/XSA by the processing-system configuration.
set pl_clock_mhz 100.0
if {[info exists ::env(LLM_FCLK_MHZ)]} {
    set pl_clock_mhz $::env(LLM_FCLK_MHZ)
}
if {![string is double -strict $pl_clock_mhz] || $pl_clock_mhz <= 0.0} {
    error "LLM_FCLK_MHZ must be a positive number, got '$pl_clock_mhz'"
}
puts "Configuring PL FCLK0 at ${pl_clock_mhz} MHz"
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $pl_clock_mhz \
] $ps7

set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]
set interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_gp0_interconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $interconnect
set accelerator [create_bd_cell -type module -reference llm_axi_queued_ip bcuda_accelerator_0]
set ddr_bridge [create_bd_cell -type module -reference memory_io_axi3_bridge_ip ddr_bridge_0]

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_gp0_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_gp0_interconnect/M00_AXI] [get_bd_intf_pins bcuda_accelerator_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins ddr_bridge_0/M_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins axi_gp0_interconnect/ACLK] \
    [get_bd_pins axi_gp0_interconnect/S00_ACLK] \
    [get_bd_pins axi_gp0_interconnect/M00_ACLK] \
    [get_bd_pins bcuda_accelerator_0/aclk] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins ddr_bridge_0/aclk] \
    [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins axi_gp0_interconnect/ARESETN] \
    [get_bd_pins axi_gp0_interconnect/S00_ARESETN] \
    [get_bd_pins axi_gp0_interconnect/M00_ARESETN] \
    [get_bd_pins bcuda_accelerator_0/aresetn]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins ddr_bridge_0/aresetn]

connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_valid] [get_bd_pins ddr_bridge_0/cmd_valid]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_ready] [get_bd_pins ddr_bridge_0/cmd_ready]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_write] [get_bd_pins ddr_bridge_0/cmd_write]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_addr] [get_bd_pins ddr_bridge_0/cmd_addr]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_len] [get_bd_pins ddr_bridge_0/cmd_len]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_wdata] [get_bd_pins ddr_bridge_0/cmd_wdata]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_cmd_wstrb] [get_bd_pins ddr_bridge_0/cmd_wstrb]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_write_valid] [get_bd_pins ddr_bridge_0/write_valid]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_write_ready] [get_bd_pins ddr_bridge_0/write_ready]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_write_data] [get_bd_pins ddr_bridge_0/write_data]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_write_strb] [get_bd_pins ddr_bridge_0/write_strb]
connect_bd_net [get_bd_pins bcuda_accelerator_0/mem_write_last] [get_bd_pins ddr_bridge_0/write_last]
connect_bd_net [get_bd_pins ddr_bridge_0/read_valid] [get_bd_pins bcuda_accelerator_0/mem_read_valid]
connect_bd_net [get_bd_pins ddr_bridge_0/read_ready] [get_bd_pins bcuda_accelerator_0/mem_read_ready]
connect_bd_net [get_bd_pins ddr_bridge_0/read_data] [get_bd_pins bcuda_accelerator_0/mem_read_data]
connect_bd_net [get_bd_pins ddr_bridge_0/read_last] [get_bd_pins bcuda_accelerator_0/mem_read_last]
connect_bd_net [get_bd_pins ddr_bridge_0/debug_arlen] [get_bd_pins bcuda_accelerator_0/mem_debug_arlen]
connect_bd_net [get_bd_pins ddr_bridge_0/debug_arsize] [get_bd_pins bcuda_accelerator_0/mem_debug_arsize]
connect_bd_net [get_bd_pins ddr_bridge_0/debug_rbeats] [get_bd_pins bcuda_accelerator_0/mem_debug_rbeats]
connect_bd_net [get_bd_pins ddr_bridge_0/debug_rresp] [get_bd_pins bcuda_accelerator_0/mem_debug_rresp]

set accelerator_segment [lindex [get_bd_addr_segs -of_objects [get_bd_intf_pins bcuda_accelerator_0/s_axi]] 0]
if {$accelerator_segment eq ""} {
    error "Unable to locate the accelerator AXI-Lite address segment"
}
assign_bd_address -offset 0x43C80000 -range 64K \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] $accelerator_segment
set ddr_segment [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]
if {$ddr_segment eq ""} {
    error "Unable to locate the PYNQ-Z2 HP0 DDR address segment"
}
assign_bd_address -offset 0x00000000 -range 512M \
    -target_address_space [get_bd_addr_spaces ddr_bridge_0/m_axi] $ddr_segment

validate_bd_design
save_bd_design
make_wrapper -files [get_files [file join $project_dir "budgetcuda_board.srcs" "sources_1" "bd" "bcuda_ps_pl" "bcuda_ps_pl.bd"]] -top
add_files -norecurse [file join $project_dir "budgetcuda_board.gen" "sources_1" "bd" "bcuda_ps_pl" "hdl" "bcuda_ps_pl_wrapper.v"]
set_property top bcuda_ps_pl_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "Created PS/PL wrapper project at $project_dir"
puts "Control register base: 0x43C80000"
