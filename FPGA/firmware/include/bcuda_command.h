#ifndef BCUDA_COMMAND_H
#define BCUDA_COMMAND_H

#include <stdint.h>

#include "bcuda_graph.h"
#include "bcuda_protocol.h"

int bcuda_handle_alloc(const BcudaAllocRequest *req, BcudaAllocResponse *res);
int bcuda_handle_write(const BcudaWriteRequest *req, const uint8_t *payload, BcudaWriteResponse *res);
int bcuda_handle_read(const BcudaReadRequest *req, BcudaReadResponse *res, uint8_t *payload_out);
int bcuda_handle_execute(BcudaGraphRuntime *runtime, const BcudaExecuteRequest *req, const uint8_t *descriptor, BcudaExecuteResponse *res);
int bcuda_handle_free(const BcudaFreeRequest *req, BcudaFreeResponse *res);

#endif
