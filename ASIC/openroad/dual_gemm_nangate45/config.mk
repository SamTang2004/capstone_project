export DESIGN_NAME = transformer_gemm_scheduler
export PLATFORM = nangate45

export BCUDA_PROJECT_HOME ?= /workspace
export BCUDA_RTL_HOME ?= $(BCUDA_PROJECT_HOME)/FPGA/hardware/rtl
export BCUDA_ASIC_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/openroad/dual_gemm_nangate45

# This implementation boundary retains both 8x8 MAC arrays and the complete
# GEMM scheduler/writeback logic.  Scratchpads are intentionally depth-scaled
# because Nangate45 has no compiled SRAM views; the full-top configuration
# separately inventories the production-sized memories.
export VERILOG_FILES = \
  $(BCUDA_RTL_HOME)/pe_int8_mac.sv \
  $(BCUDA_RTL_HOME)/weight_vector_decode.sv \
  $(BCUDA_RTL_HOME)/matrix_array_8x8.sv \
  $(BCUDA_RTL_HOME)/gemm_8x8_tile_core.sv \
  $(BCUDA_RTL_HOME)/gemm_requantize_writer.sv \
  $(BCUDA_RTL_HOME)/transformer_gemm_scheduler.sv

export VERILOG_DEFINES = -D BCUDA_ASIC
export SYNTH_HDL_FRONTEND = slang
export VERILOG_TOP_PARAMS = MAX_K 16 MAX_N 8 MAX_W4_CACHE_VECTORS 8
export SYNTH_MEMORY_MAX_BITS = 4096
export SDC_FILE = $(BCUDA_ASIC_HOME)/constraint.sdc

export ABC_AREA = 0
export CORE_UTILIZATION = 45
export CORE_ASPECT_RATIO = 1.0
export CORE_MARGIN = 20
export PLACE_DENSITY = 0.55
# Avoid an upstream FLUTE crash observed in the timing-driven global-placement
# rebuffer pass. Timing repair still runs after placement and after routing.
export GPL_TIMING_DRIVEN = 0
export TNS_END_PERCENT = 100
export SYNTH_REPEATABLE_BUILD = 1
