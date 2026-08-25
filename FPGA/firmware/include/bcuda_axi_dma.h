#ifndef BCUDA_AXI_DMA_H
#define BCUDA_AXI_DMA_H

#include <stdint.h>

#include "bcuda_buffer.h"

#define BCUDA_PL_DEFAULT_REGISTER_BASE 0x43C80000u

enum {
    BCUDA_PL_REG_CONTROL = 0x00u,
    BCUDA_PL_REG_STATUS = 0x04u,
    BCUDA_PL_REG_A_BASE = 0x08u,
    BCUDA_PL_REG_B_BASE = 0x0cu,
    BCUDA_PL_REG_C_BASE = 0x10u,
    BCUDA_PL_REG_A_STRIDE = 0x14u,
    BCUDA_PL_REG_B_STRIDE = 0x18u,
    BCUDA_PL_REG_C_STRIDE = 0x1cu,
    BCUDA_PL_REG_M = 0x20u,
    BCUDA_PL_REG_N = 0x24u,
    BCUDA_PL_REG_K = 0x28u,
    BCUDA_PL_REG_TILE_SIZE = 0x2cu,
    BCUDA_PL_REG_ARRAY_COUNT = 0x30u,
    BCUDA_PL_REG_RQ_MULTIPLIER = 0x34u,
    BCUDA_PL_REG_RQ_RIGHT_SHIFT = 0x38u,
    BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT = 0x3cu,
    BCUDA_PL_REG_CONV_CYCLES_TOTAL = 0x40u,
    BCUDA_PL_REG_CONV_CYCLES_ZERO_A = 0x44u,
    BCUDA_PL_REG_CONV_CYCLES_LOAD_A = 0x48u,
    BCUDA_PL_REG_CONV_CYCLES_ZERO_B = 0x4cu,
    BCUDA_PL_REG_CONV_CYCLES_LOAD_B = 0x50u,
    BCUDA_PL_REG_CONV_CYCLES_BIAS = 0x54u,
    BCUDA_PL_REG_CONV_CYCLES_COMPUTE = 0x58u,
    BCUDA_PL_REG_CONV_CYCLES_ACCUMULATE = 0x5cu,
    BCUDA_PL_REG_CONV_CYCLES_WRITE = 0x60u,
    BCUDA_PL_REG_CONV_LOADER_ADDRESS = 0x64u,
    BCUDA_PL_REG_CONV_LOADER_REQUEST = 0x68u,
    BCUDA_PL_REG_CONV_LOADER_RECEIVE = 0x6cu,
    BCUDA_PL_REG_CONV_CACHE_HITS = 0x70u,
    BCUDA_PL_REG_CONV_CACHE_MISSES = 0x74u,
    BCUDA_PL_REG_CONV_FILL_CYCLES = 0x78u,
    BCUDA_PL_REG_CONV_BYTES_DELIVERED = 0x7cu,
    BCUDA_PL_REG_GEMM_LOAD_A_LO = 0x48u, BCUDA_PL_REG_GEMM_LOAD_A_HI = 0x4cu,
    BCUDA_PL_REG_GEMM_LOAD_B_LO = 0x50u, BCUDA_PL_REG_GEMM_LOAD_B_HI = 0x54u,
    BCUDA_PL_REG_GEMM_COMPUTE_LO = 0x58u, BCUDA_PL_REG_GEMM_COMPUTE_HI = 0x5cu,
    BCUDA_PL_REG_GEMM_WRITE_LO = 0x60u, BCUDA_PL_REG_GEMM_WRITE_HI = 0x64u,
    BCUDA_PL_REG_GEMM_LOAD_B_WEIGHT_LO = 0xccu, BCUDA_PL_REG_GEMM_LOAD_B_WEIGHT_HI = 0xd0u,
    BCUDA_PL_REG_GEMM_LOAD_B_BIAS_LO = 0xd4u, BCUDA_PL_REG_GEMM_LOAD_B_BIAS_HI = 0xd8u,
    BCUDA_PL_REG_GEMM_W4_CACHE_HITS = 0xdcu, BCUDA_PL_REG_GEMM_W4_CACHE_MISSES = 0xe0u,
    BCUDA_PL_REG_DEBUG_CONTROL = 0x80u,
    BCUDA_PL_REG_TOTAL_CYCLES_LO = 0x84u, BCUDA_PL_REG_TOTAL_CYCLES_HI = 0x88u,
    BCUDA_PL_REG_BUSY_CYCLES_LO = 0x8cu, BCUDA_PL_REG_BUSY_CYCLES_HI = 0x90u,
    BCUDA_PL_REG_MEMORY_STALL_LO = 0x94u, BCUDA_PL_REG_MEMORY_STALL_HI = 0x98u,
    BCUDA_PL_REG_INPUT_STALL_LO = 0x9cu, BCUDA_PL_REG_INPUT_STALL_HI = 0xa0u,
    BCUDA_PL_REG_OUTPUT_STALL_LO = 0xa4u, BCUDA_PL_REG_OUTPUT_STALL_HI = 0xa8u,
    BCUDA_PL_REG_LAST_OP_LO = 0xacu, BCUDA_PL_REG_LAST_OP_HI = 0xb0u,
    BCUDA_PL_REG_MAX_OP_LO = 0xb4u, BCUDA_PL_REG_MAX_OP_HI = 0xb8u,
    BCUDA_PL_REG_COMPLETED_OPS = 0xbcu, BCUDA_PL_REG_DEBUG_STATUS = 0xc0u,
    BCUDA_PL_REG_TRACE_EVENT = 0xc4u, BCUDA_PL_REG_TRACE_DATA = 0xc8u
};

enum { BCUDA_DEBUG_ENABLE = 1u, BCUDA_DEBUG_STEP = 2u, BCUDA_DEBUG_FREEZE = 4u, BCUDA_DEBUG_CLEAR = 8u };

typedef struct {
    uint64_t total_cycles, busy_cycles, memory_stall_cycles, input_stall_cycles, output_stall_cycles;
    uint64_t last_op_cycles, max_op_cycles;
    uint64_t gemm_load_a_cycles, gemm_load_b_cycles, gemm_compute_cycles, gemm_write_cycles;
    uint64_t gemm_load_b_weight_cycles, gemm_load_b_bias_cycles;
    uint32_t gemm_w4_cache_hit_tiles, gemm_w4_cache_miss_tiles;
    uint32_t completed_ops;
    uint8_t active_op, error_latched, trace_valid, trace_event;
    uint32_t trace_data;
} BcudaTransformerDebugSnapshot;

enum {
    BCUDA_PL_CONTROL_START = 0x00000001u,
    BCUDA_PL_CONTROL_REQUANTIZE = 0x00000002u,
    BCUDA_PL_CONTROL_CONV2D = 0x00000004u,
    BCUDA_PL_CONTROL_ACTIVATION = 0x00000008u,
    BCUDA_PL_CONTROL_LEAKY_RELU = 0x00000010u,
    BCUDA_PL_CONTROL_MAXPOOL = 0x00000020u,
    BCUDA_PL_CONTROL_UPSAMPLE = 0x00000040u,
    BCUDA_PL_CONTROL_CONCAT = 0x00000080u,
    BCUDA_PL_CONTROL_AVGPOOL = 0x00000200u,
    BCUDA_PL_CONTROL_CONV_PRESLICED_WEIGHTS = 0x00000400u,
    BCUDA_PL_CONTROL_TRANSFORMER_WEIGHT_INT4 = 0x00000800u,
    BCUDA_PL_CONTROL_TRANSFORMER_REQUANT_PACK = 0x00001000u,
    BCUDA_PL_CONTROL_TRANSFORMER_B_PACK = 0x00002000u,
    BCUDA_PL_CONTROL_TRANSFORMER_B_PACK_TRANSPOSE = 0x00004000u,
    BCUDA_PL_CONTROL_TRANSFORMER_PACK_SOURCE_ROW_MAJOR = 0x00008000u,
    BCUDA_PL_CONTROL_TRANSFORMER_GELU = 0x00010000u,
    BCUDA_PL_CONTROL_TRANSFORMER_RESIDUAL = 0x00020000u,
    BCUDA_PL_CONTROL_TRANSFORMER_LAYERNORM = 0x00040000u,
    BCUDA_PL_CONTROL_TRANSFORMER_SOFTMAX = 0x00080000u,
    BCUDA_PL_CONTROL_TRANSFORMER_FUSED_REQUANT = 0x00100000u,
    BCUDA_PL_CONTROL_TRANSFORMER_OUTPUT_LAYOUT_ALT = 0x00200000u,
    BCUDA_PL_CONTROL_TRANSFORMER_OUTPUT_DYNAMIC_B = 0x00400000u,
    BCUDA_PL_CONTROL_HEATVIT_TOKEN_PRUNE = 0x00800000u,
    BCUDA_PL_STATUS_BUSY = 0x00000001u,
    BCUDA_PL_STATUS_DONE = 0x00000002u,
    BCUDA_PL_STATUS_ERROR = 0x00000004u,
    BCUDA_PL_TILE_DIM = 16u,
    BCUDA_PL_TILE_ELEMENTS = 256u,
    BCUDA_PL_ARRAY_COUNT = 1u
};

enum {
    BCUDA_PL_TRANSFORMER_TILE_DIM = 8u,
    BCUDA_TRANSFORMER_WEIGHT_INT8 = 0u,
    BCUDA_TRANSFORMER_WEIGHT_INT4 = 1u
};

typedef struct {
    uintptr_t register_base;
    uint32_t timeout_cycles;
    uint32_t last_control_word;
    uintptr_t last_source_address;
    uint32_t last_transfer_bytes;
    uint32_t last_tile_size;
    uint32_t last_array_count;
    uint32_t last_m_tiles;
    uint32_t last_n_tiles;
    uint32_t last_k_tiles;
    uint32_t last_n_groups;
    uint32_t last_schedule_steps;
    uint32_t last_accumulator_entries;
    uintptr_t last_memory_base;
    uint32_t last_memory_bytes;
    uint32_t last_memory_io_flags;
    uint32_t conv_cycles_total;
    uint32_t conv_cycles_zero_a;
    uint32_t conv_cycles_load_a;
    uint32_t conv_cycles_zero_b;
    uint32_t conv_cycles_load_b;
    uint32_t conv_cycles_bias;
    uint32_t conv_cycles_compute;
    uint32_t conv_cycles_accumulate;
    uint32_t conv_cycles_write;
    uint32_t conv_loader_address_cycles;
    uint32_t conv_loader_request_cycles;
    uint32_t conv_loader_receive_cycles;
    uint32_t conv_cache_hits;
    uint32_t conv_cache_misses;
    uint32_t conv_fill_cycles;
    uint32_t conv_bytes_delivered;
} BcudaAxiDma;

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles);
int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_submit_transformer_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *packed_A, BcudaBuffer *packed_B, BcudaBuffer *C, uint32_t weight_int4);
int bcuda_axi_dma_submit_transformer_gemm_requant(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaRequantParams *params, BcudaBuffer *packed_A, BcudaBuffer *packed_B, BcudaBuffer *bias, BcudaBuffer *packed_C, uint32_t weight_int4);
int bcuda_axi_dma_submit_transformer_requantize_pack(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *packed_output, uint32_t rows, uint32_t columns);
int bcuda_axi_dma_submit_transformer_b_pack(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *packed_output, uint32_t reduction, uint32_t outputs, uint32_t transpose, uint32_t source_row_major);
int bcuda_axi_dma_submit_transformer_elementwise(BcudaAxiDma *dma, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *output, uint32_t rows, uint32_t columns, uint32_t residual, uint32_t lhs_shift, uint32_t rhs_shift);
int bcuda_axi_dma_submit_transformer_rowwise(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *gamma, BcudaBuffer *beta, BcudaBuffer *output, uint32_t rows, uint32_t columns, uint32_t softmax, uint32_t epsilon_term);
int bcuda_axi_dma_submit_heatvit_token_prune(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *output, uint32_t input_tokens, uint32_t output_tokens, uint32_t features, uint32_t weight_int4);
int bcuda_axi_dma_submit_requantize(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_conv2d(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *bias, BcudaBuffer *output);
int bcuda_axi_dma_submit_activation(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_maxpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_avgpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_upsample(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_concat(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *input2, BcudaBuffer *output);
int bcuda_axi_dma_submit_tile_load(BcudaAxiDma *dma, const BcudaTileLoadDesc *desc, BcudaBuffer *source);
int bcuda_axi_dma_submit_tiled_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaTiledGemmConfig *config, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_configure_accumulator(BcudaAxiDma *dma, const BcudaAccumulatorConfig *config);
int bcuda_axi_dma_configure_memory_io(BcudaAxiDma *dma, const BcudaMemoryIoConfig *config);
int bcuda_axi_dma_poll_done(BcudaAxiDma *dma);
int bcuda_axi_dma_debug_control(BcudaAxiDma *dma, uint32_t flags);
int bcuda_axi_dma_read_transformer_debug(BcudaAxiDma *dma, BcudaTransformerDebugSnapshot *snapshot);

#endif
