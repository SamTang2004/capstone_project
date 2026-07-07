#ifndef BCUDA_PROTOCOL_H
#define BCUDA_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define BCUDA_MAGIC 0x42435544u
#define BCUDA_VERSION 1u

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
    BC_LAYOUT_NHWC = 3
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
    BC_OP_CONV2D_GEMM = 3
};

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

typedef struct {
    uint32_t model_desc_buffer;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t flags;
    uint32_t reserved;
} BcudaGraphExecDesc;
#pragma pack(pop)

int bcuda_validate_header(const BcudaHeader *header, uint32_t expected_command);
void bcuda_init_response(BcudaResponseHeader *response, const BcudaHeader *request, uint32_t status, uint32_t header_bytes, uint32_t payload_bytes);

#endif
