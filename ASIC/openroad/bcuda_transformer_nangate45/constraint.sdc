current_design bcuda_transformer_ddr_top

set clk_name core_clock
set clk_port_name aclk
set clk_period 10.0
set clk_io_pct 0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set clk_io_name vclk_$clk_name
create_clock -name $clk_io_name -period $clk_period

# Initial block-level assumptions. A packaged ASIC integration should replace
# these with delays derived from the selected bus fabric and memory PHY.
set_input_delay [expr $clk_period * $clk_io_pct] \
  -clock $clk_io_name [all_inputs -no_clocks]
set_output_delay [expr $clk_period * $clk_io_pct] \
  -clock $clk_io_name [all_outputs]

set_false_path -from [get_ports aresetn]
