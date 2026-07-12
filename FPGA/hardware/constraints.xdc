###############################################################################
# Budget-CUDA tiled systolic DDR timing constraints
#
# Target board family: Zynq-7020 class, e.g. PYNQ-Z2/Zybo-style devices.
# The 500 MHz target is intentionally aggressive for this RTL and device class.
# It gives Vivado a hard optimization target for synthesis/place/route while the
# actual deployed PL clock can be relaxed later after timing closure analysis.
###############################################################################

# Main PL fabric clock. 500 MHz target => 2.000 ns period.
create_clock -name aclk -period 2.000 [get_ports aclk]
set_clock_uncertainty 0.100 [get_clocks aclk]

# Reset is externally asserted and only used as a control/reset path. Do not let
# reset release dominate datapath timing reports.
set_false_path -from [get_ports aresetn]

# Top-level AXI-Lite and memory command ports are temporary standalone
# integration pins in this prototype flow. Timing is dominated by the PL clock;
# board-level input/output delay constraints belong in the PS/PL wrapper XDC.
