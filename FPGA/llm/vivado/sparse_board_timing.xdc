# Keep the same explicit margin on both compute and host clocks.
set_clock_uncertainty -setup 0.300 [get_clocks {clk_fpga_0 clk_fpga_1}]
