# Copy this file for the target board and set all PS7 configuration values
# exported by that board's Vivado preset. Do not use the FPGA part alone as a
# substitute for DDR and MIO electrical configuration.
#
# Example (the real value is board-specific):
# set_property -dict [list CONFIG.PCW_UIPARAM_DDR_PARTNO {<part>} ...] \
#     [get_bd_cells processing_system7_0]

error "Replace board_preset.template.tcl with the target board's PS7 DDR/MIO preset"
