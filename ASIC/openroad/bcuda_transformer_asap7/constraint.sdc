current_design bcuda_transformer_ddr_top

# ASAP7 Liberty files use picoseconds.  This is the FPGA design's 100 MHz
# block-level target expressed as 10,000 ps.
set clk_period 10000.000
create_clock -name core_clock -period $clk_period [get_ports aclk]
create_clock -name vclk_core_clock -period $clk_period

set_input_delay 2000.000 -clock vclk_core_clock [all_inputs -no_clocks]
set_output_delay 2000.000 -clock vclk_core_clock [all_outputs]
set_clock_uncertainty 100.000 [get_clocks core_clock]
set_false_path -from [get_ports aresetn]
