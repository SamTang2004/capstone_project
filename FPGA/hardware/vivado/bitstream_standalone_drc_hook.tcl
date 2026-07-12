# Temporary standalone-RTL bitstream hook.
#
# The standalone PL top exposes control and memory-integration ports as
# top-level RTL ports while the project is still below the full Zynq
# PS/block-design wrapper. These signals are intended to become internal PL
# nets, not package pins. Do not assign random PACKAGE_PIN locations for them.
# For this standalone checkpoint/bitstream flow, downgrade the physical-I/O DRCs
# only.
#
# Remove this hook once the design is packaged in a block design or when a real
# board-level XDC assigns every top-level port to safe pins.
set_property SEVERITY Warning [get_drc_checks {NSTD-1 UCIO-1}]
