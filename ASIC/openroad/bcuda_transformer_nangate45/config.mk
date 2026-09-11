export DESIGN_NAME = bcuda_transformer_ddr_top
export PLATFORM = nangate45

# Docker mounts the project at /workspace.  The WSL runner overrides this one
# root variable, keeping the design configuration portable between engines.
export BCUDA_PROJECT_HOME ?= /workspace
export BCUDA_RTL_HOME ?= $(BCUDA_PROJECT_HOME)/FPGA/hardware/rtl
export BCUDA_ASIC_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/openroad/bcuda_transformer_nangate45

# Keep this list aligned with FPGA/hardware/vivado/final_bitstream_sources.tcl.
# The Xilinx interface and PS/PL wrappers are intentionally excluded: this
# block models the accelerator core at its AXI-Lite and memory-command boundary.
export VERILOG_FILES = \
  $(BCUDA_RTL_HOME)/axi_lite_control.sv \
  $(BCUDA_RTL_HOME)/pe_int8_mac.sv \
  $(BCUDA_RTL_HOME)/weight_vector_decode.sv \
  $(BCUDA_RTL_HOME)/matrix_array_8x8.sv \
  $(BCUDA_RTL_HOME)/gemm_8x8_tile_core.sv \
  $(BCUDA_RTL_HOME)/gemm_requantize_writer.sv \
  $(BCUDA_RTL_HOME)/transformer_gemm_scheduler.sv \
  $(BCUDA_RTL_HOME)/requantize_ddr.sv \
  $(BCUDA_RTL_HOME)/requantize_pack_int8.sv \
  $(BCUDA_RTL_HOME)/activation_b_pack_int8.sv \
  $(BCUDA_RTL_HOME)/gelu_int8.sv \
  $(BCUDA_RTL_HOME)/residual_add_int8.sv \
  $(BCUDA_RTL_HOME)/iterative_unsigned_divider.sv \
  $(BCUDA_RTL_HOME)/transformer_elementwise_ddr.sv \
  $(BCUDA_RTL_HOME)/layernorm_int8.sv \
  $(BCUDA_RTL_HOME)/softmax_int8.sv \
  $(BCUDA_RTL_HOME)/rowwise_norm_softmax_engine.sv \
  $(BCUDA_RTL_HOME)/transformer_rowwise_ddr.sv \
  $(BCUDA_RTL_HOME)/transformer_debug_counters.sv \
  $(BCUDA_RTL_HOME)/transformer_control_status.sv \
  $(BCUDA_RTL_HOME)/heatvit_token_scorer_4lane.sv \
  $(BCUDA_RTL_HOME)/heatvit_token_pruner_ddr.sv \
  $(BCUDA_RTL_HOME)/memory_io_driver.sv \
  $(BCUDA_RTL_HOME)/bcuda_transformer_ddr_top.sv

export VERILOG_DEFINES = -D BCUDA_ASIC
export SYNTH_HDL_FRONTEND = slang
# Stop after reporting the production memory inventory instead of accidentally
# exploding the 589-Kbit store into hundreds of thousands of flip-flops.
export SYNTH_MEMORY_MAX_BITS = 4096
export SDC_FILE = $(BCUDA_ASIC_HOME)/constraint.sdc

# Match the routed FPGA target first (100 MHz), so area/power comparisons do
# not accidentally trade away the existing throughput target.
export ABC_AREA = 0
export CORE_UTILIZATION = 45
export CORE_ASPECT_RATIO = 1.0
export CORE_MARGIN = 20
export PLACE_DENSITY = 0.55
export TNS_END_PERCENT = 100
export SYNTH_REPEATABLE_BUILD = 1
