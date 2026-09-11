# ASAP7 Liberty timing units are picoseconds: these values are equivalent to
# the Nangate45 experiment's 10 ns clock, 2 ns I/O delay, and 0.1 ns margin.
create_clock -name core_clk -period 10000.000 [get_ports clk]
set_input_delay 2000.000 -clock core_clk [all_inputs -no_clocks]
set_output_delay 2000.000 -clock core_clk [all_outputs]
set_clock_uncertainty 100.000 [get_clocks core_clk]
set_false_path -from [get_ports rst_n]
