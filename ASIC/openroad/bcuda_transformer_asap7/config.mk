export DESIGN_NAME = bcuda_transformer_ddr_top
export DESIGN_NICKNAME = bcuda_transformer_asap7_sram
export PLATFORM = asap7

export BCUDA_PROJECT_HOME ?= /workspace
export BCUDA_RTL_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/rtl
export BCUDA_ASIC_HOME ?= $(BCUDA_PROJECT_HOME)/ASIC/openroad/bcuda_transformer_asap7

export VERILOG_FILES = \
  $(BCUDA_RTL_HOME)/asap7_fakeram_blackboxes.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_8.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_16.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_32.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_40.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_64.sv \
  $(BCUDA_RTL_HOME)/bcuda_asap7_sram_1rw_masked_64.sv \
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
export SYNTH_MEMORY_MAX_BITS = 4096
export SDC_FILE = $(BCUDA_ASIC_HOME)/constraint.sdc
export PDN_TCL = $(BCUDA_ASIC_HOME)/pdn.tcl

export ADDITIONAL_LEFS = \
  $(PLATFORM_DIR)/lef/fakeram7_128x64.lef \
  $(PLATFORM_DIR)/lef/fakeram_512x8.lef \
  $(PLATFORM_DIR)/lef/fakeram7_256x32.lef
export ADDITIONAL_LIBS = \
  $(PLATFORM_DIR)/lib/NLDM/fakeram7_128x64.lib \
  $(PLATFORM_DIR)/lib/NLDM/fakeram_512x8.lib \
  $(PLATFORM_DIR)/lib/NLDM/fakeram7_256x32.lib

export CORNER = TC
export ASAP7_USE_VT = RVT
export ABC_AREA = 0
export CORE_UTILIZATION = 25
export CORE_ASPECT_RATIO = 1.0
export CORE_MARGIN = 15
export PLACE_DENSITY = 0.35
export MACRO_PLACE_HALO = 2 2
export GPL_TIMING_DRIVEN = 0
export TNS_END_PERCENT = 100
export SYNTH_REPEATABLE_BUILD = 1
