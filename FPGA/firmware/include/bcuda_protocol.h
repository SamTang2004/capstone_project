#ifndef BCUDA_PROTOCOL_H
#define BCUDA_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define BCUDA_MAGIC 0x42435544u
#define BCUDA_VERSION 1u
#define BCUDA_MODEL_MAGIC 0x444D4342u
#define BCUDA_MODEL_VERSION 1u
#define BCUDA_ACTIVATION_LEAKY_RELU 0x00000001u
#define BCUDA_ACTIVATION_LEAKY_SHIFT_SHIFT 8u
#define BCUDA_ACTIVATION_LEAKY_SHIFT_MASK 0x00000700u
#define BCUDA_POOL_AVERAGE 0x00000001u
#define BCUDA_CONV_PRESLICED_WEIGHTS 0x00000001u
#define BCUDA_DIAGNOSTICS_MAGIC 0x47444342u
#define BCUDA_DIAGNOSTICS_VERSION 4u
#define BCUDA_DIAG_TIMING 0x00000001u
#define BCUDA_DIAG_MEMORY 0x00000002u
#define BCUDA_DIAG_OPERATIONS 0x00000004u
#define BCUDA_DIAG_UTILIZATION 0x00000008u
#define BCUDA_DIAG_ALL 0x0000000fu

enum {
    BC_CMD_ALLOC = 1,
    BC_CMD_WRITE = 2,
    BC_CMD_READ = 3,
    BC_CMD_EXECUTE = 4,
    BC_CMD_FREE = 5
};

enum {
    BC_DTYPE_INT8 = 1,
    BC_DTYPE_INT16 = 2,
    BC_DTYPE_INT32 = 3
};

enum {
    BC_LAYOUT_ROW_MAJOR = 1,
    BC_LAYOUT_NCHW = 2,
    BC_LAYOUT_NHWC = 3,
    BC_LAYOUT_CONV_WEIGHT_TILED_INT8 = 4,
    BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 = 5,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8 = 6,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4 = 7,
    BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8 = 8
};

enum {
    BC_USAGE_WEIGHT = 1,
    BC_USAGE_ACTIVATION = 2,
    BC_USAGE_WORKSPACE = 3,
    BC_USAGE_OUTPUT = 4,
    BC_USAGE_DESCRIPTOR = 5
};

enum {
    BC_OP_GEMM = 1,
    BC_OP_GRAPH = 2,
    BC_OP_CONV2D_GEMM = 3,
    BC_OP_ACTIVATION = 4,
    BC_OP_POOL = 5,
    BC_OP_UPSAMPLE = 6,
    BC_OP_CONCAT = 7,
    BC_OP_RESHAPE = 8,
    BC_OP_DETECT_DECODE = 9,
    BC_OP_NMS = 10,
    BC_OP_REQUANTIZE = 11,
    BC_OP_LAYERNORM = 12,
    BC_OP_GELU = 13,
    BC_OP_SOFTMAX = 14,
    BC_OP_RESIDUAL_ADD = 15,
    BC_OP_ACTIVATION_B_PACK = 16,
    BC_OP_TOKEN_CONCAT = 17,
    BC_OP_TOKEN_EMBEDDING = 18
};

/* Transformer primitive flags. Existing opcodes and descriptor layout remain ABI-compatible. */
#define BCUDA_RESIDUAL_LHS_SHIFT_MASK 0x00000007u
#define BCUDA_RESIDUAL_RHS_SHIFT_SHIFT 4u
#define BCUDA_RESIDUAL_RHS_SHIFT_MASK 0x00000070u
#define BCUDA_GEMM_DYNAMIC_B_TRANSPOSE 0x00000001u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t command;
    uint32_t request_id;
    uint32_t header_bytes;
    uint32_t payload_bytes;
    uint32_t flags;
    uint32_t checksum;
} BcudaHeader;

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t command;
    uint32_t request_id;
    uint32_t status;
    uint32_t header_bytes;
    uint32_t payload_bytes;
    uint32_t error_code;
    uint32_t reserved;
} BcudaResponseHeader;

typedef struct {
    uint32_t dtype;
    uint32_t rank;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t dim3;
    uint32_t layout;
    uint32_t reserved;
} BcudaTensorDesc;

typedef struct {
    BcudaHeader header;
    uint64_t size_bytes;
    BcudaTensorDesc tensor;
    uint32_t usage;
    uint32_t alignment;
} BcudaAllocRequest;

typedef struct {
    BcudaResponseHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t size_bytes;
} BcudaAllocResponse;

typedef struct {
    BcudaHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t offset_bytes;
    uint64_t write_bytes;
} BcudaWriteRequest;

typedef struct {
    BcudaResponseHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t written_bytes;
} BcudaWriteResponse;

typedef struct {
    BcudaHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t offset_bytes;
    uint64_t read_bytes;
} BcudaReadRequest;

typedef struct {
    BcudaResponseHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t read_bytes;
} BcudaReadResponse;

typedef struct {
    BcudaHeader header;
    uint32_t op_type;
    uint32_t descriptor_bytes;
} BcudaExecuteRequest;

typedef struct {
    BcudaResponseHeader header;
    uint32_t op_type;
    uint32_t execution_status;
    uint64_t elapsed_cycles;
} BcudaExecuteResponse;

typedef struct {
    BcudaHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
} BcudaFreeRequest;

typedef struct {
    BcudaResponseHeader header;
    uint32_t buffer_id;
    uint32_t reserved;
} BcudaFreeResponse;

typedef struct {
    uint32_t A_buffer;
    uint32_t B_buffer;
    uint32_t C_buffer;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t A_dtype;
    uint32_t B_dtype;
    uint32_t C_dtype;
    uint32_t A_layout;
    uint32_t B_layout;
    uint32_t C_layout;
    uint32_t flags;
    uint32_t reserved;
} BcudaGemmDesc;

/* INT32 accumulator to INT8 activation conversion.  right_shift is 0..31. */
typedef struct {
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t element_count;
    int32_t multiplier;
    uint32_t right_shift;
    int32_t input_zero_point;
    int32_t output_zero_point;
    uint32_t flags;
} BcudaRequantDesc;

/* Contents of the INT32 scale buffer referenced by a REQUANTIZE layer. */
typedef struct {
    int32_t multiplier;
    uint32_t right_shift;
    int32_t input_zero_point;
    int32_t output_zero_point;
} BcudaRequantParams;

typedef struct {
    uint32_t buffer_id;
    uint32_t matrix_select;
    uint64_t base_offset_bytes;
    uint32_t row_stride_bytes;
    uint32_t tile_rows;
    uint32_t tile_cols;
    uint32_t dtype;
    uint32_t flags;
} BcudaTileLoadDesc;

typedef struct {
    uint32_t tile_size;
    uint32_t array_count;
    uint32_t flags;
    uint32_t reserved;
} BcudaTiledGemmConfig;

typedef struct {
    uint32_t tile_entries;
    uint32_t clear_on_first_pass;
    uint32_t flags;
    uint32_t reserved;
} BcudaAccumulatorConfig;

typedef struct {
    uint64_t memory_base;
    uint32_t memory_bytes;
    uint32_t flags;
} BcudaMemoryIoConfig;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t layer_capacity;
    uint32_t layer_count;
    uint32_t record_bytes;
    uint32_t clock_hz;
    uint32_t peak_operations_per_cycle;
    uint64_t total_elapsed_ns;
    uint64_t arena_bytes_used;
} BcudaDiagnosticsHeader;

typedef struct {
    uint32_t layer_index;
    uint32_t op_type;
    uint32_t status;
    uint32_t reserved0;
    uint64_t elapsed_ns;
    uint64_t cycles;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t weight_bytes;
    uint64_t operations;
    uint64_t throughput_ops_per_second;
    uint32_t utilization_ppm;
    uint32_t gemm_bias_elapsed_ns;
    uint64_t hardware_total_cycles;
    uint64_t hardware_busy_cycles;
    uint64_t hardware_memory_stall_cycles;
    uint64_t hardware_input_stall_cycles;
    uint64_t hardware_output_stall_cycles;
    uint64_t hardware_last_op_cycles;
    uint64_t hardware_max_op_cycles;
    uint64_t hardware_gemm_load_a_cycles;
    uint64_t hardware_gemm_load_b_cycles;
    uint64_t hardware_gemm_compute_cycles;
    uint64_t hardware_gemm_write_cycles;
    uint32_t hardware_completed_ops;
    uint32_t hardware_status;
} BcudaDiagnosticsRecord;

typedef struct {
    uint32_t model_desc_buffer;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t flags;
    uint32_t diagnostics_buffer;
} BcudaGraphExecDesc;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_layers;
    uint32_t layer_desc_offset;
    uint32_t num_buffers;
    uint32_t buffer_table_offset;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t activation_arena_a;
    uint32_t activation_arena_b;
    uint32_t workspace_buffer;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
} BcudaModelDesc;

typedef struct {
    uint32_t buffer_id;
    uint32_t dtype;
    uint32_t layout;
    uint32_t usage;
    uint32_t size_bytes;
    uint32_t rank;
    uint32_t dim0;
    uint32_t dim1;
} BcudaModelBufferDesc;

typedef struct {
    uint32_t op_type;
    uint32_t input0;
    uint32_t input1;
    uint32_t input2;
    uint32_t output;
    uint32_t weight;
    uint32_t bias;
    uint32_t scale;
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    uint32_t out_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t groups;
    uint32_t flags;
    uint32_t reserved;
} BcudaLayerDesc;
#pragma pack(pop)

int bcuda_validate_header(const BcudaHeader *header, uint32_t expected_command);
void bcuda_init_response(BcudaResponseHeader *response, const BcudaHeader *request, uint32_t status, uint32_t header_bytes, uint32_t payload_bytes);

#endif
