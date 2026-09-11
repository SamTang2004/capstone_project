export DESIGN_NAME = transformer_gemm_scheduler
export DESIGN_NICKNAME = dual_gemm_asap7_sram
export PLATFORM = asap7

export BCUDA_PROJECT_HOME ?= /workspace
export BCUDA_RTL_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/rtl
export BCUDA_ASIC_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/openroad/dual_gemm_asap7

# ASIC deployment RTL uses explicit ASAP7 SRAM macro wrappers.  The platform
# black-box model supplies the macro module declaration to synthesis.
export VERILOG_FILES = \
  $(PLATFORM_DIR)/verilog/fakeram7_128x64.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_64.sv \
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

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram7_128x64.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/NLDM/fakeram7_128x64.lib
export MACRO_PLACE_HALO = 5 5

# ASAP7 defaults to its best-case corner. Use typical RVT libraries for this
# architectural comparison.
export CORNER = TC
export ASAP7_USE_VT = RVT
export ABC_AREA = 0
export CORE_UTILIZATION = 30
export CORE_ASPECT_RATIO = 1.0
export CORE_MARGIN = 10
export PLACE_DENSITY = 0.40
export GPL_TIMING_DRIVEN = 0
export TNS_END_PERCENT = 100
export SYNTH_REPEATABLE_BUILD = 1
