set root [file normalize [file join [file dirname [info script]] ..]]
set hw [file normalize [file join $root .. hardware]]
set mhz 200
if {[llength $argv]} {set mhz [lindex $argv 0]}
if {$mhz ni {100 200}} {error "Supported board build clocks: 100 or 200 MHz"}
set project_dir [file join $root build sparse_board${mhz}]
set output_dir [file join $root build sparse_release${mhz}]
file mkdir $output_dir
set_param board.repoPaths [list [file join $hw board_files XilinxBoardStore boards]]
create_project sparse_board $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
foreach name {llm_bitmap_finder64 llm_dot_product_lane llm_dot_core8 llm_activation_block_map llm_tiled_activation8 llm_sparse_board_memory llm_sparse_board_control} {
    add_files -norecurse [file join $root rtl ${name}.sv]
}
add_files -norecurse [file join $root rtl llm_sparse_board_ip.v]
add_files -fileset constrs_1 -norecurse [file join $root vivado sparse_board_timing.xdc]
update_compile_order -fileset sources_1
create_bd_design sparse_ps_pl
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $mhz CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {100}] $ps
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_host
set bus [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_gp0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1} CONFIG.S00_HAS_REGSLICE {1} CONFIG.M00_HAS_REGSLICE {1}] $bus
create_bd_cell -type module -reference llm_sparse_board_ip sparse_accelerator
set pc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 host_protocol]
set_property -dict [list CONFIG.SI_PROTOCOL {AXI3} CONFIG.MI_PROTOCOL {AXI4LITE}] $pc
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins host_protocol/S_AXI]
connect_bd_intf_net [get_bd_intf_pins host_protocol/M_AXI] [get_bd_intf_pins axi_gp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_gp0/M00_AXI] [get_bd_intf_pins sparse_accelerator/S_AXI]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_gp0/M00_ACLK] [get_bd_pins sparse_accelerator/aclk] [get_bd_pins rst_ps7/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK1] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins axi_gp0/ACLK] [get_bd_pins axi_gp0/S00_ACLK] [get_bd_pins rst_host/slowest_sync_clk] [get_bd_pins host_protocol/aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7/ext_reset_in] [get_bd_pins rst_host/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7/peripheral_aresetn] [get_bd_pins axi_gp0/M00_ARESETN] [get_bd_pins sparse_accelerator/aresetn]
connect_bd_net [get_bd_pins rst_host/peripheral_aresetn] [get_bd_pins axi_gp0/ARESETN] [get_bd_pins axi_gp0/S00_ARESETN] [get_bd_pins host_protocol/aresetn]
set seg [lindex [get_bd_addr_segs -of_objects [get_bd_intf_pins sparse_accelerator/s_axi]] 0]
if {$seg eq ""} {error "Missing AXI segment"}
assign_bd_address -offset 0x43C90000 -range 64K -target_address_space [get_bd_addr_spaces processing_system7_0/Data] $seg
validate_bd_design
save_bd_design
set bd [get_files [file join $project_dir sparse_board.srcs sources_1 bd sparse_ps_pl sparse_ps_pl.bd]]
make_wrapper -files $bd -top
add_files -norecurse [file join $project_dir sparse_board.gen sources_1 bd sparse_ps_pl hdl sparse_ps_pl_wrapper.v]
set_property top sparse_ps_pl_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
open_run synth_1
report_utilization -hierarchical -file [file join $output_dir utilization_synth.rpt]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -hierarchical -file [file join $output_dir utilization_impl.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file [file join $output_dir timing_impl.rpt]
report_drc -file [file join $output_dir drc.rpt]
report_route_status -file [file join $output_dir route_status.rpt]
report_clock_interaction -file [file join $output_dir clock_interaction.rpt]
report_cdc -details -file [file join $output_dir cdc.rpt]
report_bus_skew -warn_on_violation -file [file join $output_dir bus_skew.rpt]
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$wns<0 || $whs<0} {error "Board timing failed WNS=$wns WHS=$whs; no bitstream exported"}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Bitstream failed"}
file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] sparse_ps_pl_wrapper.bit] [file join $output_dir sparse_pynq_z2.bit]
file copy -force [file join $project_dir sparse_board.gen sources_1 bd sparse_ps_pl hw_handoff sparse_ps_pl.hwh] [file join $output_dir sparse_pynq_z2.hwh]
write_hw_platform -fixed -include_bit -force [file join $output_dir sparse_pynq_z2.xsa]
puts "SPARSE_BOARD_RELEASE_PASS MHz=$mhz WNS=$wns WHS=$whs OUTPUT=$output_dir"
