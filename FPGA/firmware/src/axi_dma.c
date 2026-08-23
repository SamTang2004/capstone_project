#include "bcuda_axi_dma.h"

#include <stddef.h>

#include "bcuda_status.h"

static uint32_t bcuda_ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static volatile uint32_t *bcuda_pl_regs(BcudaAxiDma *dma) {
    return dma->register_base == 0 ? 0 : (volatile uint32_t *)dma->register_base;
}

static void bcuda_pl_write(BcudaAxiDma *dma, uint32_t offset, uint32_t value) {
    volatile uint32_t *regs = bcuda_pl_regs(dma);
    if (regs != 0) {
        regs[offset / sizeof(uint32_t)] = value;
    }
}

static uint32_t bcuda_pl_read(BcudaAxiDma *dma, uint32_t offset) {
    volatile uint32_t *regs = bcuda_pl_regs(dma);
    return regs == 0 ? 0 : regs[offset / sizeof(uint32_t)];
}

static uint64_t bcuda_pl_read_u64(BcudaAxiDma *dma, uint32_t low_offset) {
    uint32_t high1, low, high2;
    do { high1 = bcuda_pl_read(dma, low_offset + 4u); low = bcuda_pl_read(dma, low_offset); high2 = bcuda_pl_read(dma, low_offset + 4u); } while (high1 != high2);
    return ((uint64_t)high2 << 32) | low;
}

int bcuda_axi_dma_debug_control(BcudaAxiDma *dma, uint32_t flags) {
    if (dma == 0 || (flags & ~(BCUDA_DEBUG_ENABLE | BCUDA_DEBUG_STEP | BCUDA_DEBUG_FREEZE | BCUDA_DEBUG_CLEAR)) != 0) return BC_STATUS_GENERAL_ERROR;
    bcuda_pl_write(dma, BCUDA_PL_REG_DEBUG_CONTROL, flags);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_read_transformer_debug(BcudaAxiDma *dma, BcudaTransformerDebugSnapshot *snapshot) {
    if (dma == 0 || snapshot == 0) return BC_STATUS_GENERAL_ERROR;
    snapshot->total_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_TOTAL_CYCLES_LO);
    snapshot->busy_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_BUSY_CYCLES_LO);
    snapshot->memory_stall_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_MEMORY_STALL_LO);
    snapshot->input_stall_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_INPUT_STALL_LO);
    snapshot->output_stall_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_OUTPUT_STALL_LO);
    snapshot->last_op_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_LAST_OP_LO);
    snapshot->max_op_cycles = bcuda_pl_read_u64(dma, BCUDA_PL_REG_MAX_OP_LO);
    snapshot->completed_ops = bcuda_pl_read(dma, BCUDA_PL_REG_COMPLETED_OPS);
    uint32_t status = bcuda_pl_read(dma, BCUDA_PL_REG_DEBUG_STATUS);
    snapshot->active_op = status & 0xffu; snapshot->error_latched = (status >> 8) & 1u; snapshot->trace_valid = (status >> 9) & 1u;
    snapshot->trace_event = bcuda_pl_read(dma, BCUDA_PL_REG_TRACE_EVENT) & 0xffu;
    snapshot->trace_data = bcuda_pl_read(dma, BCUDA_PL_REG_TRACE_DATA);
    return BC_STATUS_OK;
}

static void bcuda_read_conv_profile(BcudaAxiDma *dma) {
    dma->conv_cycles_total = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_TOTAL);
    dma->conv_cycles_zero_a = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_ZERO_A);
    dma->conv_cycles_load_a = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_LOAD_A);
    dma->conv_cycles_zero_b = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_ZERO_B);
    dma->conv_cycles_load_b = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_LOAD_B);
    dma->conv_cycles_bias = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_BIAS);
    dma->conv_cycles_compute = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_COMPUTE);
    dma->conv_cycles_accumulate = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_ACCUMULATE);
    dma->conv_cycles_write = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CYCLES_WRITE);
    dma->conv_loader_address_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_LOADER_ADDRESS);
    dma->conv_loader_request_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_LOADER_REQUEST);
    dma->conv_loader_receive_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_LOADER_RECEIVE);
    dma->conv_cache_hits = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CACHE_HITS);
    dma->conv_cache_misses = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_CACHE_MISSES);
    dma->conv_fill_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_FILL_CYCLES);
    dma->conv_bytes_delivered = bcuda_pl_read(dma, BCUDA_PL_REG_CONV_BYTES_DELIVERED);
}

static uint32_t bcuda_stride_bytes(const BcudaBuffer *buffer, uint32_t fallback) {
    if (buffer != 0 && buffer->rank >= 2 && buffer->shape[1] != 0) {
        uint32_t element_bytes = buffer->dtype == BC_DTYPE_INT32 ? 4u : 1u;
        return buffer->shape[1] * element_bytes;
    }
    return fallback;
}

static int bcuda_validate_tiled_gemm(const BcudaGemmDesc *desc, const BcudaBuffer *A, const BcudaBuffer *B, const BcudaBuffer *C) {
    uint64_t a_required;
    uint64_t b_required;
    uint64_t c_required;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t c_stride;

    // The PL controller operates on 16-wide tiles.  Host/compiler padding is
    // required for edge tiles so the PS never exposes an undersized CMA row.
    if (desc->M == 0 || desc->N == 0 || desc->K == 0 ||
        (desc->M % BCUDA_PL_TILE_DIM) != 0 ||
        (desc->N % BCUDA_PL_TILE_DIM) != 0 ||
        (desc->K % BCUDA_PL_TILE_DIM) != 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (A->dtype != BC_DTYPE_INT8 || B->dtype != BC_DTYPE_INT8 || C->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->A_layout != BC_LAYOUT_ROW_MAJOR || desc->B_layout != BC_LAYOUT_ROW_MAJOR || desc->C_layout != BC_LAYOUT_ROW_MAJOR ||
        A->layout != BC_LAYOUT_ROW_MAJOR || B->layout != BC_LAYOUT_ROW_MAJOR || C->layout != BC_LAYOUT_ROW_MAJOR) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    a_stride = bcuda_stride_bytes(A, desc->K);
    b_stride = bcuda_stride_bytes(B, desc->N);
    c_stride = bcuda_stride_bytes(C, desc->N * sizeof(uint32_t));
    if (a_stride < desc->K || b_stride < desc->N || c_stride < desc->N * sizeof(uint32_t) ||
        (c_stride & 0x3u) != 0u) return BC_STATUS_SHAPE_MISMATCH;

    a_required = (uint64_t)(desc->M - 1u) * a_stride + desc->K;
    b_required = (uint64_t)(desc->K - 1u) * b_stride + desc->N;
    c_required = (uint64_t)(desc->M - 1u) * c_stride + (uint64_t)desc->N * sizeof(uint32_t);
    if (a_required > A->size_bytes || b_required > B->size_bytes || c_required > C->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    return BC_STATUS_OK;
}

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles) {
    if (dma == 0) {
        return;
    }
    dma->register_base = register_base;
    dma->timeout_cycles = timeout_cycles;
    dma->last_control_word = 0;
    dma->last_source_address = 0;
    dma->last_transfer_bytes = 0;
    dma->last_tile_size = 0;
    dma->last_array_count = 0;
    dma->last_m_tiles = 0;
    dma->last_n_tiles = 0;
    dma->last_k_tiles = 0;
    dma->last_n_groups = 0;
    dma->last_schedule_steps = 0;
    dma->last_accumulator_entries = 0;
    dma->last_memory_base = 0;
    dma->last_memory_bytes = 0;
    dma->last_memory_io_flags = 0;
    dma->conv_cycles_total = 0; dma->conv_cycles_zero_a = 0; dma->conv_cycles_load_a = 0;
    dma->conv_cycles_zero_b = 0; dma->conv_cycles_load_b = 0; dma->conv_cycles_bias = 0;
    dma->conv_cycles_compute = 0; dma->conv_cycles_accumulate = 0; dma->conv_cycles_write = 0;
    dma->conv_loader_address_cycles = 0; dma->conv_loader_request_cycles = 0; dma->conv_loader_receive_cycles = 0;
    dma->conv_cache_hits = 0; dma->conv_cache_misses = 0; dma->conv_fill_cycles = 0; dma->conv_bytes_delivered = 0;
}

int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    if (dma == 0 || desc == 0 || A == 0 || B == 0 || C == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (desc->A_dtype != BC_DTYPE_INT8 || desc->B_dtype != BC_DTYPE_INT8 || desc->C_dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    int status = bcuda_validate_tiled_gemm(desc, A, B, C);
    if (status != BC_STATUS_OK) {
        return status;
    }

    uint32_t a_stride = bcuda_stride_bytes(A, desc->K);
    uint32_t b_stride = bcuda_stride_bytes(B, desc->N);
    uint32_t c_stride = bcuda_stride_bytes(C, desc->N * sizeof(uint32_t));

    if (dma->register_base != 0 &&
        (A->physical_address > 0xffffffffu || B->physical_address > 0xffffffffu || C->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = A->physical_address;
    dma->last_transfer_bytes = (uint32_t)(A->size_bytes + B->size_bytes + C->size_bytes);
    dma->last_tile_size = BCUDA_PL_TILE_DIM;
    dma->last_array_count = BCUDA_PL_ARRAY_COUNT;
    dma->last_control_word = BCUDA_PL_CONTROL_START;

    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)A->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)B->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)C->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, a_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, b_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, c_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, desc->M);
        bcuda_pl_write(dma, BCUDA_PL_REG_N, desc->N);
        bcuda_pl_write(dma, BCUDA_PL_REG_K, desc->K);
        bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, BCUDA_PL_TILE_DIM);
        bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, BCUDA_PL_ARRAY_COUNT);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, BCUDA_PL_CONTROL_START);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_transformer_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *packed_A, BcudaBuffer *packed_B, BcudaBuffer *C, uint32_t weight_int4) {
    uint64_t a_required, b_required, c_required;
    uint32_t m_tiles, n_tiles, weight_vector_bytes, c_stride, control;
    if (dma == 0 || desc == 0 || packed_A == 0 || packed_B == 0 || C == 0) return BC_STATUS_GENERAL_ERROR;
    if (desc->A_dtype != BC_DTYPE_INT8 || desc->B_dtype != BC_DTYPE_INT8 || desc->C_dtype != BC_DTYPE_INT32 ||
        packed_A->dtype != BC_DTYPE_INT8 || packed_B->dtype != BC_DTYPE_INT8 || C->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->M == 0 || desc->N == 0 || desc->K == 0 || desc->K > 1024u || weight_int4 > 1u) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    m_tiles = bcuda_ceil_div_u32(desc->M, BCUDA_PL_TRANSFORMER_TILE_DIM);
    n_tiles = bcuda_ceil_div_u32(desc->N, BCUDA_PL_TRANSFORMER_TILE_DIM);
    weight_vector_bytes = weight_int4 ? 4u : 8u;
    c_stride = bcuda_stride_bytes(C, desc->N * sizeof(uint32_t));
    a_required = (uint64_t)m_tiles * desc->K * 8u;
    b_required = (uint64_t)n_tiles * desc->K * weight_vector_bytes;
    c_required = (uint64_t)(desc->M - 1u) * c_stride + (uint64_t)desc->N * sizeof(uint32_t);
    if (c_stride < desc->N * sizeof(uint32_t) || (c_stride & 3u) != 0u ||
        a_required > packed_A->size_bytes || b_required > packed_B->size_bytes || c_required > C->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (dma->register_base != 0 && (packed_A->physical_address > 0xffffffffu || packed_B->physical_address > 0xffffffffu || C->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    control = BCUDA_PL_CONTROL_START | (weight_int4 ? BCUDA_PL_CONTROL_TRANSFORMER_WEIGHT_INT4 : 0u);
    dma->last_source_address = packed_A->physical_address;
    dma->last_transfer_bytes = (uint32_t)(a_required + b_required + c_required);
    dma->last_tile_size = BCUDA_PL_TRANSFORMER_TILE_DIM;
    dma->last_array_count = BCUDA_PL_ARRAY_COUNT;
    dma->last_control_word = control;
    dma->last_m_tiles = m_tiles;
    dma->last_n_tiles = n_tiles;
    dma->last_k_tiles = desc->K;
    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)packed_A->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)packed_B->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)C->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, desc->K * 8u);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, desc->K * weight_vector_bytes);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, c_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, desc->M);
        bcuda_pl_write(dma, BCUDA_PL_REG_N, desc->N);
        bcuda_pl_write(dma, BCUDA_PL_REG_K, desc->K);
        bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, BCUDA_PL_TRANSFORMER_TILE_DIM);
        bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, BCUDA_PL_ARRAY_COUNT);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, control);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_transformer_requantize_pack(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *packed_output, uint32_t rows, uint32_t columns) {
    uint32_t input_stride;
    uint64_t input_required, output_required;
    if (dma == 0 || desc == 0 || input == 0 || packed_output == 0) return BC_STATUS_GENERAL_ERROR;
    if (input->dtype != BC_DTYPE_INT32 || packed_output->dtype != BC_DTYPE_INT8) return BC_STATUS_UNSUPPORTED_DTYPE;
    if (rows == 0 || columns == 0 || rows > 0xffffu || columns > 0xffffu || desc->right_shift > 31u ||
        desc->input_zero_point != 0 || desc->element_count != rows * columns) return BC_STATUS_SHAPE_MISMATCH;
    input_stride = bcuda_stride_bytes(input, columns * sizeof(int32_t));
    input_required = (uint64_t)(rows - 1u) * input_stride + (uint64_t)columns * sizeof(int32_t);
    output_required = (uint64_t)bcuda_ceil_div_u32(rows, BCUDA_PL_TRANSFORMER_TILE_DIM) * columns * 8u;
    if (input_stride < columns * sizeof(int32_t) || (input_stride & 3u) != 0u ||
        input_required > input->size_bytes || output_required > packed_output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if (dma->register_base != 0 && (input->physical_address > 0xffffffffu || packed_output->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    dma->last_source_address = input->physical_address;
    dma->last_transfer_bytes = (uint32_t)(input_required + output_required);
    dma->last_tile_size = BCUDA_PL_TRANSFORMER_TILE_DIM;
    dma->last_array_count = BCUDA_PL_ARRAY_COUNT;
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_TRANSFORMER_REQUANT_PACK;
    dma->last_m_tiles = bcuda_ceil_div_u32(rows, BCUDA_PL_TRANSFORMER_TILE_DIM);
    dma->last_n_tiles = columns;
    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)packed_output->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, input_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, rows);
        bcuda_pl_write(dma, BCUDA_PL_REG_N, columns);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, (uint32_t)desc->multiplier);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, desc->right_shift);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, (uint32_t)desc->output_zero_point);
        bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, BCUDA_PL_TRANSFORMER_TILE_DIM);
        bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, BCUDA_PL_ARRAY_COUNT);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_transformer_b_pack(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *packed_output, uint32_t reduction, uint32_t outputs, uint32_t transpose, uint32_t source_row_major) {
    uint64_t input_required, output_required;
    uint32_t source_rows, source_features;
    if (dma == 0 || input == 0 || packed_output == 0) return BC_STATUS_GENERAL_ERROR;
    if (input->dtype != BC_DTYPE_INT8 || packed_output->dtype != BC_DTYPE_INT8) return BC_STATUS_UNSUPPORTED_DTYPE;
    if (reduction == 0 || outputs == 0 || reduction > 0xffffu || outputs > 0xffffu || transpose > 1u || source_row_major > 1u) return BC_STATUS_SHAPE_MISMATCH;
    source_rows = transpose ? outputs : reduction;
    source_features = transpose ? reduction : outputs;
    input_required = source_row_major ? (uint64_t)source_rows * source_features :
        (uint64_t)bcuda_ceil_div_u32(source_rows, BCUDA_PL_TRANSFORMER_TILE_DIM) * source_features * 8u;
    output_required = (uint64_t)bcuda_ceil_div_u32(outputs, BCUDA_PL_TRANSFORMER_TILE_DIM) * reduction * 8u;
    if (input->layout != (source_row_major ? BC_LAYOUT_ROW_MAJOR : BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8) ||
        packed_output->layout != (source_row_major ? BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 : BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8) ||
        input_required > input->size_bytes || output_required > packed_output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if (dma->register_base != 0 && (input->physical_address > 0xffffffffu || packed_output->physical_address > 0xffffffffu)) return BC_STATUS_SHAPE_MISMATCH;
    dma->last_source_address = input->physical_address;
    dma->last_transfer_bytes = (uint32_t)(input_required + output_required);
    dma->last_tile_size = BCUDA_PL_TRANSFORMER_TILE_DIM;
    dma->last_array_count = BCUDA_PL_ARRAY_COUNT;
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_TRANSFORMER_B_PACK |
        (transpose ? BCUDA_PL_CONTROL_TRANSFORMER_B_PACK_TRANSPOSE : 0u) |
        (source_row_major ? BCUDA_PL_CONTROL_TRANSFORMER_PACK_SOURCE_ROW_MAJOR : 0u);
    dma->last_m_tiles = bcuda_ceil_div_u32(source_rows, BCUDA_PL_TRANSFORMER_TILE_DIM);
    dma->last_n_tiles = bcuda_ceil_div_u32(outputs, BCUDA_PL_TRANSFORMER_TILE_DIM);
    dma->last_k_tiles = reduction;
    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)packed_output->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, reduction);
        bcuda_pl_write(dma, BCUDA_PL_REG_N, outputs);
        bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, BCUDA_PL_TRANSFORMER_TILE_DIM);
        bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, BCUDA_PL_ARRAY_COUNT);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_transformer_elementwise(BcudaAxiDma *dma, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *output, uint32_t rows, uint32_t columns, uint32_t residual, uint32_t lhs_shift, uint32_t rhs_shift) {
    uint64_t bytes;
    if (dma == 0 || input0 == 0 || output == 0) return BC_STATUS_GENERAL_ERROR;
    if (input0->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 || (residual && (input1 == 0 || input1->dtype != BC_DTYPE_INT8))) return BC_STATUS_UNSUPPORTED_DTYPE;
    bytes = (uint64_t)rows * columns;
    if (rows == 0 || columns == 0 || rows > 0xffffu || columns > 0xffffu || bytes > input0->size_bytes || bytes > output->size_bytes ||
        (residual && (bytes > input1->size_bytes || lhs_shift > 7u || rhs_shift > 7u))) return BC_STATUS_SHAPE_MISMATCH;
    if (input0->layout != output->layout || (residual && input1->layout != output->layout)) return BC_STATUS_SHAPE_MISMATCH;
    dma->last_source_address = input0->physical_address; dma->last_transfer_bytes = (uint32_t)(bytes * (residual ? 3u : 2u));
    dma->last_control_word = BCUDA_PL_CONTROL_START | (residual ? BCUDA_PL_CONTROL_TRANSFORMER_RESIDUAL : BCUDA_PL_CONTROL_TRANSFORMER_GELU);
    dma->last_m_tiles = rows; dma->last_n_tiles = columns;
    if (dma->register_base != 0) {
        if (input0->physical_address > 0xffffffffu || output->physical_address > 0xffffffffu || (residual && input1->physical_address > 0xffffffffu)) return BC_STATUS_SHAPE_MISMATCH;
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE,(uint32_t)input0->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE,residual?(uint32_t)input1->physical_address:0u);bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE,(uint32_t)output->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_M,rows);bcuda_pl_write(dma, BCUDA_PL_REG_N,columns);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT,lhs_shift);bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT,rhs_shift);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL,dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_transformer_rowwise(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *gamma, BcudaBuffer *beta, BcudaBuffer *output, uint32_t rows, uint32_t columns, uint32_t softmax, uint32_t epsilon_term) {
    uint64_t input_bytes=(uint64_t)rows*columns,output_bytes;
    if(dma==0||input==0||output==0)return BC_STATUS_GENERAL_ERROR;
    if(input->dtype!=BC_DTYPE_INT8||output->dtype!=BC_DTYPE_INT8)return BC_STATUS_UNSUPPORTED_DTYPE;
    if(rows==0||columns==0||rows>0xffffu||columns>0xffffu||(softmax&&columns>256u))return BC_STATUS_SHAPE_MISMATCH;
    output_bytes=softmax?input_bytes:((uint64_t)(rows+7u)/8u)*columns*8u;
    if(input->layout!=BC_LAYOUT_ROW_MAJOR||output->layout!=(softmax?BC_LAYOUT_ROW_MAJOR:BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8)||input_bytes>input->size_bytes||output_bytes>output->size_bytes)return BC_STATUS_SHAPE_MISMATCH;
    if(!softmax&&(gamma==0||beta==0||gamma->dtype!=BC_DTYPE_INT16||beta->dtype!=BC_DTYPE_INT16||gamma->size_bytes<(uint64_t)columns*2u||beta->size_bytes<(uint64_t)columns*2u))return BC_STATUS_SHAPE_MISMATCH;
    dma->last_source_address=input->physical_address;dma->last_transfer_bytes=(uint32_t)(input_bytes+output_bytes+(softmax?0u:columns*4u));dma->last_control_word=BCUDA_PL_CONTROL_START|(softmax?BCUDA_PL_CONTROL_TRANSFORMER_SOFTMAX:BCUDA_PL_CONTROL_TRANSFORMER_LAYERNORM);dma->last_m_tiles=rows;dma->last_n_tiles=columns;
    if(dma->register_base!=0){if(input->physical_address>0xffffffffu||output->physical_address>0xffffffffu||(!softmax&&(gamma->physical_address>0xffffffffu||beta->physical_address>0xffffffffu)))return BC_STATUS_SHAPE_MISMATCH;
      bcuda_pl_write(dma,BCUDA_PL_REG_CONTROL,0);bcuda_pl_write(dma,BCUDA_PL_REG_A_BASE,(uint32_t)input->physical_address);bcuda_pl_write(dma,BCUDA_PL_REG_B_BASE,softmax?0:(uint32_t)gamma->physical_address);bcuda_pl_write(dma,BCUDA_PL_REG_B_STRIDE,softmax?0:(uint32_t)beta->physical_address);bcuda_pl_write(dma,BCUDA_PL_REG_C_BASE,(uint32_t)output->physical_address);bcuda_pl_write(dma,BCUDA_PL_REG_M,rows);bcuda_pl_write(dma,BCUDA_PL_REG_N,columns);bcuda_pl_write(dma,BCUDA_PL_REG_RQ_MULTIPLIER,epsilon_term);bcuda_pl_write(dma,BCUDA_PL_REG_CONTROL,dma->last_control_word);}
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_requantize(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *output) {
    if (dma == 0 || desc == 0 || input == 0 || output == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (input->dtype != BC_DTYPE_INT32 || output->dtype != BC_DTYPE_INT8) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->element_count == 0 || desc->right_shift > 31u || desc->input_zero_point != 0 ||
        (uint64_t)desc->element_count * sizeof(int32_t) > input->size_bytes || desc->element_count > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (dma->register_base != 0 && (input->physical_address > 0xffffffffu || output->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = input->physical_address;
    dma->last_transfer_bytes = desc->element_count * (uint32_t)(sizeof(int32_t) + sizeof(int8_t));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_REQUANTIZE;
    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, desc->element_count);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, (uint32_t)desc->multiplier);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, desc->right_shift);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, (uint32_t)desc->output_zero_point);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_conv2d(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *bias, BcudaBuffer *output) {
    uint64_t reduction;
    uint64_t required_weight_bytes;
    if (dma == 0 || layer == 0 || input == 0 || weight == 0 || output == 0) return BC_STATUS_GENERAL_ERROR;
    if (input->dtype != BC_DTYPE_INT8 || weight->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT32 ||
        layer->n != 1u || layer->groups != 1u || layer->c == 0u || layer->h == 0u || layer->w == 0u || layer->out_c == 0u ||
        layer->out_h == 0u || layer->out_w == 0u || layer->kernel_h == 0u || layer->kernel_h > 16u || layer->kernel_w == 0u ||
        layer->kernel_w > 16u || layer->stride_h == 0u || layer->stride_w == 0u || layer->stride_h > 255u || layer->stride_w > 255u ||
        layer->pad_h > 255u || layer->pad_w > 255u || (bias != 0 && (bias->dtype != BC_DTYPE_INT32 || bias->size_bytes < (uint64_t)layer->out_c * sizeof(int32_t)))) return BC_STATUS_SHAPE_MISMATCH;
    reduction = (uint64_t)layer->c * layer->kernel_h * layer->kernel_w;
    required_weight_bytes = (layer->flags & BCUDA_CONV_PRESLICED_WEIGHTS) != 0u
        ? ((layer->out_c + 15u) / 16u) * ((reduction + 15u) / 16u) * 256u
        : (uint64_t)layer->out_c * reduction;
    if (weight->size_bytes < required_weight_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if (dma->register_base == 0) return BC_STATUS_EXECUTION_FAILED;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)weight->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->c);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->w);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->out_c);
    bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, bias == 0 ? 0u : (uint32_t)bias->physical_address);
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_CONV2D;
    if ((layer->flags & BCUDA_CONV_PRESLICED_WEIGHTS) != 0u) dma->last_control_word |= BCUDA_PL_CONTROL_CONV_PRESLICED_WEIGHTS;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_activation(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    uint32_t flags;
    if (dma == 0 || layer == 0 || input == 0 || output == 0 || input->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 || input->size_bytes != output->size_bytes || input->size_bytes == 0 || dma->register_base == 0) return BC_STATUS_SHAPE_MISMATCH;
    flags = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_ACTIVATION;
    if ((layer->flags & BCUDA_ACTIVATION_LEAKY_RELU) != 0u) flags |= BCUDA_PL_CONTROL_LEAKY_RELU;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, (uint32_t)input->size_bytes);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, (layer->flags & BCUDA_ACTIVATION_LEAKY_SHIFT_MASK) >> BCUDA_ACTIVATION_LEAKY_SHIFT_SHIFT);
    dma->last_control_word = flags;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, flags);
    return BC_STATUS_OK;
}

static int bcuda_pl_validate_int8_tensor_op(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *output) {
    if (dma == 0 || input == 0 || output == 0 || dma->register_base == 0 || input->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 ||
        input->physical_address > 0xffffffffu || output->physical_address > 0xffffffffu) return BC_STATUS_SHAPE_MISMATCH;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_maxpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    if (layer == 0 || bcuda_pl_validate_int8_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 ||
        layer->out_h == 0 || layer->out_w == 0 || layer->kernel_h == 0 || layer->kernel_h > 16 || layer->kernel_w == 0 || layer->kernel_w > 16 ||
        layer->stride_h == 0 || layer->stride_h > 255 || layer->stride_w == 0 || layer->stride_w > 255 || layer->pad_h > 255 || layer->pad_w > 255 ||
        input->size_bytes < (uint64_t)layer->n * layer->c * layer->h * layer->w || output->size_bytes < (uint64_t)layer->n * layer->c * layer->out_h * layer->out_w) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h); bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8)); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_MAXPOOL;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_avgpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    if (layer == 0 || bcuda_pl_validate_int8_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 ||
        layer->out_h == 0 || layer->out_w == 0 || layer->kernel_h == 0 || layer->kernel_h > 16 || layer->kernel_w == 0 || layer->kernel_w > 16 ||
        layer->stride_h == 0 || layer->stride_h > 255 || layer->stride_w == 0 || layer->stride_w > 255 || layer->pad_h > 255 || layer->pad_w > 255 ||
        input->size_bytes < (uint64_t)layer->n * layer->c * layer->h * layer->w || output->size_bytes < (uint64_t)layer->n * layer->c * layer->out_h * layer->out_w) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h); bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8)); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_AVGPOOL;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_upsample(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    uint32_t scale_h, scale_w;
    if (layer == 0 || bcuda_pl_validate_int8_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 || layer->out_h == 0 || layer->out_w == 0 ||
        layer->out_h % layer->h != 0 || layer->out_w % layer->w != 0 || input->size_bytes < (uint64_t)layer->n * layer->c * layer->h * layer->w || output->size_bytes < (uint64_t)layer->n * layer->c * layer->out_h * layer->out_w) return BC_STATUS_SHAPE_MISMATCH;
    scale_h = layer->out_h / layer->h; scale_w = layer->out_w / layer->w;
    if (scale_h == 0 || scale_h > 255 || scale_w == 0 || scale_w > 255) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, scale_h | (scale_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_UPSAMPLE;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_concat(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *input2, BcudaBuffer *output) {
    uint32_t c0, c1, c2;
    if (layer == 0 || input0 == 0 || input1 == 0 || output == 0 || bcuda_pl_validate_int8_tensor_op(dma, input0, output) != BC_STATUS_OK || input1->dtype != BC_DTYPE_INT8 || input1->physical_address > 0xffffffffu || layer->n == 0 || layer->h == 0 || layer->w == 0 || layer->out_c == 0) return BC_STATUS_SHAPE_MISMATCH;
    c0 = (uint32_t)(input0->size_bytes / ((uint64_t)layer->n * layer->h * layer->w)); c1 = (uint32_t)(input1->size_bytes / ((uint64_t)layer->n * layer->h * layer->w));
    c2 = input2 == 0 ? 0u : (uint32_t)(input2->size_bytes / ((uint64_t)layer->n * layer->h * layer->w));
    if (c0 == 0 || c1 == 0 || c0 + c1 + c2 != layer->out_c || c1 > 65535 || c2 > 65535 || (input2 != 0 && (input2->dtype != BC_DTYPE_INT8 || input2->physical_address > 0xffffffffu)) || output->size_bytes < (uint64_t)layer->n * layer->out_c * layer->h * layer->w) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input0->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)input1->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, c0); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, c1); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, c2); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, input2 == 0 ? 0u : (uint32_t)input2->physical_address);
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_CONCAT;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_tile_load(BcudaAxiDma *dma, const BcudaTileLoadDesc *desc, BcudaBuffer *source) {
    if (dma == 0 || desc == 0 || source == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (desc->dtype != BC_DTYPE_INT8 || source->dtype != BC_DTYPE_INT8) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->tile_rows == 0 || desc->tile_cols == 0 || desc->tile_rows > 16 || desc->tile_cols > 16) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (desc->row_stride_bytes < desc->tile_cols) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint64_t last_row_offset = desc->base_offset_bytes + (uint64_t)(desc->tile_rows - 1u) * desc->row_stride_bytes;
    uint64_t final_byte = last_row_offset + desc->tile_cols;
    if (final_byte > source->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = source->physical_address + (uintptr_t)desc->base_offset_bytes;
    dma->last_transfer_bytes = desc->tile_rows * desc->tile_cols;
    dma->last_control_word = 0x2u;
    dma->last_control_word ^= (uint32_t)(dma->last_source_address & 0xffffffffu);
    dma->last_control_word ^= desc->row_stride_bytes;
    dma->last_control_word ^= (desc->tile_rows << 16) | desc->tile_cols;
    dma->last_control_word ^= (desc->matrix_select & 1u) << 31;
    dma->last_control_word ^= desc->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_tiled_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaTiledGemmConfig *config, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    if (dma == 0 || desc == 0 || config == 0 || A == 0 || B == 0 || C == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (desc->A_dtype != BC_DTYPE_INT8 || desc->B_dtype != BC_DTYPE_INT8 || desc->C_dtype != BC_DTYPE_INT32 ||
        A->dtype != BC_DTYPE_INT8 || B->dtype != BC_DTYPE_INT8 || C->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (config->tile_size != BCUDA_PL_TILE_DIM || config->array_count != BCUDA_PL_ARRAY_COUNT) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t m_tiles = bcuda_ceil_div_u32(desc->M, config->tile_size);
    uint32_t n_tiles = bcuda_ceil_div_u32(desc->N, config->tile_size);
    uint32_t k_tiles = bcuda_ceil_div_u32(desc->K, config->tile_size);
    uint32_t n_groups = bcuda_ceil_div_u32(n_tiles, config->array_count);
    uint32_t schedule_steps = m_tiles * n_groups * k_tiles;

    dma->last_tile_size = config->tile_size;
    dma->last_array_count = config->array_count;
    dma->last_m_tiles = m_tiles;
    dma->last_n_tiles = n_tiles;
    dma->last_k_tiles = k_tiles;
    dma->last_n_groups = n_groups;
    dma->last_schedule_steps = schedule_steps;
    dma->last_source_address = A->physical_address;
    dma->last_transfer_bytes = (uint32_t)(A->size_bytes + B->size_bytes + C->size_bytes);
    return bcuda_axi_dma_submit_gemm(dma, desc, A, B, C);
}

int bcuda_axi_dma_configure_accumulator(BcudaAxiDma *dma, const BcudaAccumulatorConfig *config) {
    if (dma == 0 || config == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (config->tile_entries == 0 || config->tile_entries > 256) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_accumulator_entries = config->tile_entries;
    dma->last_control_word = 0x8u;
    dma->last_control_word ^= config->tile_entries;
    dma->last_control_word ^= (config->clear_on_first_pass & 1u) << 16;
    dma->last_control_word ^= config->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_configure_memory_io(BcudaAxiDma *dma, const BcudaMemoryIoConfig *config) {
    if (dma == 0 || config == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (config->memory_base == 0 || config->memory_bytes == 0 || (config->memory_base & 0x3u) != 0 || (config->memory_bytes & 0x3u) != 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_memory_base = (uintptr_t)config->memory_base;
    dma->last_memory_bytes = config->memory_bytes;
    dma->last_memory_io_flags = config->flags;
    dma->last_control_word = 0x20u;
    dma->last_control_word ^= (uint32_t)(config->memory_base & 0xffffffffu);
    dma->last_control_word ^= config->memory_bytes;
    dma->last_control_word ^= config->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_poll_done(BcudaAxiDma *dma) {
    if (dma == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (dma->register_base == 0) {
        return dma->timeout_cycles == 0 ? BC_STATUS_EXECUTION_FAILED : BC_STATUS_OK;
    }

    for (uint32_t cycle = 0; cycle < dma->timeout_cycles; ++cycle) {
        uint32_t status = bcuda_pl_read(dma, BCUDA_PL_REG_STATUS);
        if ((status & BCUDA_PL_STATUS_ERROR) != 0u) {
            bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
            return BC_STATUS_EXECUTION_FAILED;
        }
        /* DONE marks the producer's final request; BUSY remains asserted
         * until the shared HP0 write path has drained it. */
        if ((status & (BCUDA_PL_STATUS_DONE | BCUDA_PL_STATUS_BUSY)) == BCUDA_PL_STATUS_DONE) {
            bcuda_read_conv_profile(dma);
            bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
            return BC_STATUS_OK;
        }
    }
    return BC_STATUS_EXECUTION_FAILED;
}
