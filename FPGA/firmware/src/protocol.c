#include "bcuda_protocol.h"

#include "bcuda_status.h"

int bcuda_validate_header(const BcudaHeader *header, uint32_t expected_command) {
    if (header == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (header->magic != BCUDA_MAGIC) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (header->version != BCUDA_VERSION) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (expected_command != 0 && header->command != expected_command) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (header->header_bytes < sizeof(BcudaHeader)) {
        return BC_STATUS_INVALID_COMMAND;
    }
    return BC_STATUS_OK;
}

void bcuda_init_response(BcudaResponseHeader *response, const BcudaHeader *request, uint32_t status, uint32_t header_bytes, uint32_t payload_bytes) {
    response->magic = BCUDA_MAGIC;
    response->version = BCUDA_VERSION;
    response->command = request ? request->command : 0;
    response->request_id = request ? request->request_id : 0;
    response->status = status;
    response->header_bytes = header_bytes;
    response->payload_bytes = payload_bytes;
    response->error_code = status == BC_STATUS_OK ? 0 : status;
    response->reserved = 0;
}
