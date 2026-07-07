#include "bcuda_command.h"

#include <string.h>

#include "bcuda_buffer.h"
#include "bcuda_status.h"

int bcuda_handle_alloc(const BcudaAllocRequest *req, BcudaAllocResponse *res) {
    if (req == 0 || res == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    int status = bcuda_validate_header(&req->header, BC_CMD_ALLOC);
    uint32_t buffer_id = 0;
    if (status == BC_STATUS_OK) {
        status = bcuda_alloc_buffer(req->size_bytes, req->tensor, req->usage, &buffer_id);
    }
    bcuda_init_response(&res->header, &req->header, status, sizeof(*res), 0);
    res->buffer_id = buffer_id;
    res->reserved = 0;
    res->size_bytes = status == BC_STATUS_OK ? req->size_bytes : 0;
    return status;
}

int bcuda_handle_write(const BcudaWriteRequest *req, const uint8_t *payload, BcudaWriteResponse *res) {
    if (req == 0 || res == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    int status = bcuda_validate_header(&req->header, BC_CMD_WRITE);
    if (status == BC_STATUS_OK) {
        status = bcuda_write_buffer(req->buffer_id, req->offset_bytes, payload, req->write_bytes);
    }
    bcuda_init_response(&res->header, &req->header, status, sizeof(*res), 0);
    res->buffer_id = req->buffer_id;
    res->reserved = 0;
    res->written_bytes = status == BC_STATUS_OK ? req->write_bytes : 0;
    return status;
}

int bcuda_handle_read(const BcudaReadRequest *req, BcudaReadResponse *res, uint8_t *payload_out) {
    if (req == 0 || res == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    int status = bcuda_validate_header(&req->header, BC_CMD_READ);
    if (status == BC_STATUS_OK) {
        status = bcuda_read_buffer(req->buffer_id, req->offset_bytes, payload_out, req->read_bytes);
    }
    bcuda_init_response(&res->header, &req->header, status, sizeof(*res), status == BC_STATUS_OK ? (uint32_t)req->read_bytes : 0);
    res->buffer_id = req->buffer_id;
    res->reserved = 0;
    res->read_bytes = status == BC_STATUS_OK ? req->read_bytes : 0;
    return status;
}

int bcuda_handle_execute(BcudaGraphRuntime *runtime, const BcudaExecuteRequest *req, const uint8_t *descriptor, BcudaExecuteResponse *res) {
    if (req == 0 || res == 0 || descriptor == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    int status = bcuda_validate_header(&req->header, BC_CMD_EXECUTE);
    if (status == BC_STATUS_OK) {
        if (req->op_type == BC_OP_GEMM) {
            if (req->descriptor_bytes < sizeof(BcudaGemmDesc)) {
                status = BC_STATUS_SHAPE_MISMATCH;
            } else {
                BcudaGemmDesc desc;
                memcpy(&desc, descriptor, sizeof(desc));
                status = bcuda_execute_gemm(runtime, &desc);
            }
        } else if (req->op_type == BC_OP_GRAPH) {
            if (req->descriptor_bytes < sizeof(BcudaGraphExecDesc)) {
                status = BC_STATUS_SHAPE_MISMATCH;
            } else {
                BcudaGraphExecDesc desc;
                memcpy(&desc, descriptor, sizeof(desc));
                status = bcuda_execute_graph(runtime, &desc);
            }
        } else {
            status = BC_STATUS_INVALID_COMMAND;
        }
    }
    bcuda_init_response(&res->header, &req->header, status, sizeof(*res), 0);
    res->op_type = req->op_type;
    res->execution_status = status;
    res->elapsed_cycles = 0;
    return status;
}

int bcuda_handle_free(const BcudaFreeRequest *req, BcudaFreeResponse *res) {
    if (req == 0 || res == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    int status = bcuda_validate_header(&req->header, BC_CMD_FREE);
    if (status == BC_STATUS_OK) {
        status = bcuda_free_buffer(req->buffer_id);
    }
    bcuda_init_response(&res->header, &req->header, status, sizeof(*res), 0);
    res->buffer_id = req->buffer_id;
    res->reserved = 0;
    return status;
}
