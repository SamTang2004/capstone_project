#include "bcuda_protocol_server.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>

#include "bcuda_command.h"
#include "bcuda_status.h"

static int recv_exact(int fd, void *buffer, size_t bytes) {
    uint8_t *cursor = (uint8_t *)buffer;
    while (bytes != 0) {
        ssize_t received = recv(fd, cursor, bytes, 0);
        if (received <= 0) return -1;
        cursor += (size_t)received;
        bytes -= (size_t)received;
    }
    return 0;
}

static int send_exact(int fd, const void *buffer, size_t bytes) {
    const uint8_t *cursor = (const uint8_t *)buffer;
    while (bytes != 0) {
        ssize_t sent = send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (sent <= 0) return -1;
        cursor += (size_t)sent;
        bytes -= (size_t)sent;
    }
    return 0;
}

static int send_error(int fd, const BcudaHeader *request, uint32_t status) {
    BcudaResponseHeader response;
    bcuda_init_response(&response, request, status, sizeof(response), 0);
    response.error_code = status;
    return send_exact(fd, &response, sizeof(response));
}

static int request_is_valid(const BcudaHeader *header, uint32_t bytes, uint32_t expected_header_bytes, uint32_t expected_payload_bytes) {
    return header->magic == BCUDA_MAGIC && header->version == BCUDA_VERSION &&
        header->header_bytes == expected_header_bytes && header->payload_bytes == expected_payload_bytes &&
        bytes == expected_header_bytes + expected_payload_bytes;
}

int bcuda_protocol_server_handle_client(int client_fd, BcudaGraphRuntime *runtime) {
    BcudaHeader header;
    uint8_t *body = 0;
    uint64_t total_bytes;
    int result = -1;

    if (runtime == 0 || recv_exact(client_fd, &header, sizeof(header)) != 0) return -1;
    total_bytes = (uint64_t)header.header_bytes + header.payload_bytes;
    if (header.magic != BCUDA_MAGIC || header.version != BCUDA_VERSION ||
        header.header_bytes < sizeof(header) || total_bytes > sizeof(header) + BCUDA_PROTOCOL_MAX_PAYLOAD_BYTES) {
        send_error(client_fd, &header, BC_STATUS_GENERAL_ERROR);
        return -1;
    }
    body = (uint8_t *)malloc((size_t)(total_bytes - sizeof(header)));
    if (body == 0 && total_bytes != sizeof(header)) {
        send_error(client_fd, &header, BC_STATUS_OUT_OF_MEMORY);
        return -1;
    }
    if (total_bytes != sizeof(header) && recv_exact(client_fd, body, (size_t)(total_bytes - sizeof(header))) != 0) goto done;

    if (header.command == BC_CMD_ALLOC && request_is_valid(&header, (uint32_t)total_bytes, sizeof(BcudaAllocRequest), 0)) {
        BcudaAllocRequest request; BcudaAllocResponse response;
        memcpy(&request, &header, sizeof(header)); memcpy((uint8_t *)&request + sizeof(header), body, sizeof(request) - sizeof(header));
        bcuda_handle_alloc(&request, &response);
        result = send_exact(client_fd, &response, sizeof(response));
    } else if (header.command == BC_CMD_WRITE && header.header_bytes == sizeof(BcudaWriteRequest) && header.payload_bytes <= BCUDA_PROTOCOL_MAX_PAYLOAD_BYTES) {
        BcudaWriteRequest request; BcudaWriteResponse response;
        memcpy(&request, &header, sizeof(header)); memcpy((uint8_t *)&request + sizeof(header), body, sizeof(request) - sizeof(header));
        if (request.write_bytes != header.payload_bytes) result = send_error(client_fd, &header, BC_STATUS_SHAPE_MISMATCH);
        else { bcuda_handle_write(&request, body + sizeof(request) - sizeof(header), &response); result = send_exact(client_fd, &response, sizeof(response)); }
    } else if (header.command == BC_CMD_READ && request_is_valid(&header, (uint32_t)total_bytes, sizeof(BcudaReadRequest), 0)) {
        BcudaReadRequest request; BcudaReadResponse response; uint8_t *payload;
        memcpy(&request, &header, sizeof(header)); memcpy((uint8_t *)&request + sizeof(header), body, sizeof(request) - sizeof(header));
        payload = request.read_bytes == 0 ? 0 : (uint8_t *)malloc((size_t)request.read_bytes);
        if (request.read_bytes > BCUDA_PROTOCOL_MAX_PAYLOAD_BYTES) result = send_error(client_fd, &header, BC_STATUS_SHAPE_MISMATCH);
        else if (request.read_bytes != 0 && payload == 0) result = send_error(client_fd, &header, BC_STATUS_OUT_OF_MEMORY);
        else { bcuda_handle_read(&request, &response, payload); result = send_exact(client_fd, &response, sizeof(response)); if (result == 0 && response.header.payload_bytes != 0) result = send_exact(client_fd, payload, response.header.payload_bytes); free(payload); }
    } else if (header.command == BC_CMD_EXECUTE && header.header_bytes == sizeof(BcudaExecuteRequest)) {
        BcudaExecuteRequest request; BcudaExecuteResponse response;
        memcpy(&request, &header, sizeof(header)); memcpy((uint8_t *)&request + sizeof(header), body, sizeof(request) - sizeof(header));
        if (request.descriptor_bytes != header.payload_bytes) result = send_error(client_fd, &header, BC_STATUS_SHAPE_MISMATCH);
        else { bcuda_handle_execute(runtime, &request, body + sizeof(request) - sizeof(header), &response); result = send_exact(client_fd, &response, sizeof(response)); }
    } else if (header.command == BC_CMD_FREE && request_is_valid(&header, (uint32_t)total_bytes, sizeof(BcudaFreeRequest), 0)) {
        BcudaFreeRequest request; BcudaFreeResponse response;
        memcpy(&request, &header, sizeof(header)); memcpy((uint8_t *)&request + sizeof(header), body, sizeof(request) - sizeof(header));
        bcuda_handle_free(&request, &response);
        result = send_exact(client_fd, &response, sizeof(response));
    } else {
        result = send_error(client_fd, &header, BC_STATUS_INVALID_COMMAND);
    }
done:
    free(body);
    return result;
}

int bcuda_protocol_server_run(uint16_t port, BcudaGraphRuntime *runtime) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int reuse = 1;
    struct sockaddr_in address;
    if (server_fd < 0 || runtime == 0) return -1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    memset(&address, 0, sizeof(address)); address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_ANY); address.sin_port = htons(port);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) != 0 || listen(server_fd, 8) != 0) { close(server_fd); return -1; }
    signal(SIGPIPE, SIG_IGN);
    for (;;) {
        int client_fd = accept(server_fd, 0, 0);
        if (client_fd >= 0) {
            int no_delay = 1;
            setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
            while (bcuda_protocol_server_handle_client(client_fd, runtime) == 0) {}
            close(client_fd);
        } else if (errno != EINTR) break;
    }
    close(server_fd);
    return -1;
}
