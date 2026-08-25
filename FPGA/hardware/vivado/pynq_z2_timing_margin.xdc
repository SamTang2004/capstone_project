# Reserve implementation margin beyond the PS7 clock model's generated
# jitter.  A routed image with only a few picoseconds of WNS proved unstable
# under full accelerator traffic even though static timing technically passed.
# This constraint requires place/route to retain roughly 0.15 ns more margin
# than the default 0.154 ns clock uncertainty at the same 100 MHz period.
set_clock_uncertainty -setup 0.300 [get_clocks clk_fpga_0]

