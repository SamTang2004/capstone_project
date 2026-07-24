#ifndef BCUDA_PROTOCOL_SERVER_H
#define BCUDA_PROTOCOL_SERVER_H

#include <stdint.h>

#include "bcuda_graph.h"

#define BCUDA_PROTOCOL_DEFAULT_PORT 9000u
#define BCUDA_PROTOCOL_MAX_PAYLOAD_BYTES (64u * 1024u * 1024u)

/* Serve one request on an accepted POSIX TCP socket, then close it. */
int bcuda_protocol_server_handle_client(int client_fd, BcudaGraphRuntime *runtime);

/* Bind and serve requests serially until the process is terminated. */
int bcuda_protocol_server_run(uint16_t port, BcudaGraphRuntime *runtime);

#endif
