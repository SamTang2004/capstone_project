`timescale 1ns/1ps
module memory_io_driver #(
    parameter int BURST_WORDS = 16,
    parameter int READ_QUEUE_DEPTH = 8,
    parameter int CACHE_LINES = 4
) (
    input logic clk, input logic rst_n,
    input logic invalidate_cache,
    input logic read_req_valid, output logic read_req_ready, input logic [31:0] read_req_addr,
    input logic [4:0] read_req_len, input logic read_req_word_mode,
    output logic read_data_valid, input logic read_data_ready, output logic [7:0] read_data,
    output logic read_data_last,
    output logic read_word_valid, input logic read_word_ready,
    output logic [31:0] read_word_data, output logic [3:0] read_word_keep,
    output logic read_word_last,
    input logic write_req_valid, output logic write_req_ready, input logic [31:0] write_req_addr,
    input logic [31:0] write_req_data, input logic [3:0] write_req_strb, input logic write_req_last,
    output logic mem_cmd_valid, input logic mem_cmd_ready, output logic mem_cmd_write,
    output logic [31:0] mem_cmd_addr, output logic [3:0] mem_cmd_len,
    output logic [31:0] mem_cmd_wdata, output logic [3:0] mem_cmd_wstrb,
    output logic mem_write_valid, input logic mem_write_ready,
    output logic [31:0] mem_write_data, output logic [3:0] mem_write_strb, output logic mem_write_last,
    input logic mem_read_valid, output logic mem_read_ready, input logic [31:0] mem_read_data,
    input logic mem_read_last, output logic busy, output logic error
);
    localparam int READ_PTR_W = $clog2(READ_QUEUE_DEPTH);
    localparam int SLOT_W = $clog2(CACHE_LINES);
    localparam int REF_W = $clog2(READ_QUEUE_DEPTH + 1);
    localparam int RETURN_DEPTH = 16;
    localparam int RETURN_PTR_W = $clog2(RETURN_DEPTH);
    localparam int RETURN_ID_W =
        READ_PTR_W > SLOT_W ? READ_PTR_W : SLOT_W;

    // Four small tagged slots expose independent DDR line fills while retaining
    // the accelerator's in-order byte/word response contract.
    logic [31:0] cache [0:CACHE_LINES-1][0:BURST_WORDS-1];
    logic [31:0] slot_base [0:CACHE_LINES-1];
    logic [REF_W-1:0] slot_refs [0:CACHE_LINES-1];
    logic [CACHE_LINES-1:0] slot_allocated;
    logic [CACHE_LINES-1:0] slot_valid;
    logic [CACHE_LINES-1:0] slot_cmd_pending;
    logic [CACHE_LINES-1:0] slot_fill_active;

    logic [31:0] read_addr_queue [0:READ_QUEUE_DEPTH-1];
    logic [4:0] read_len_queue [0:READ_QUEUE_DEPTH-1];
    logic read_mode_queue [0:READ_QUEUE_DEPTH-1];
    logic [SLOT_W-1:0] read_slot_queue [0:READ_QUEUE_DEPTH-1];
    logic read_mapped_queue [0:READ_QUEUE_DEPTH-1];
    logic read_direct_queue [0:READ_QUEUE_DEPTH-1];
    logic direct_pending_queue [0:READ_QUEUE_DEPTH-1];
    logic direct_ready_queue [0:READ_QUEUE_DEPTH-1];
    logic [31:0] direct_data_queue [0:READ_QUEUE_DEPTH-1];
    logic [READ_PTR_W-1:0] read_queue_head, read_queue_tail;
    logic [READ_PTR_W-1:0] map_cursor;
    logic [READ_PTR_W:0] read_queue_count;
    logic [READ_PTR_W:0] unmapped_count;
    logic [4:0] response_index;
    logic map_probe_wait, last_map_direct;

    // AXI returns same-ID bursts in issue order. Each return tag identifies
    // either a direct word request or a cache-line fill.
    logic return_direct_fifo [0:RETURN_DEPTH-1];
    logic [RETURN_ID_W-1:0] return_id_fifo [0:RETURN_DEPTH-1];
    logic [RETURN_PTR_W-1:0] return_head, return_tail;
    logic [RETURN_PTR_W:0] return_count;
    logic [3:0] fill_word_index;

    logic write_flush_pending, write_send_active, write_inflight;
    logic [31:0] write_base;
    logic [3:0] write_count, write_burst_len, write_send_index;
    logic [31:0] write_cache [0:BURST_WORDS-1];
    logic [3:0] write_strobe_cache [0:BURST_WORDS-1];

    logic request_slot_found, request_slot_hit;
    logic [SLOT_W-1:0] request_slot;
    logic pending_slot_found;
    logic [SLOT_W-1:0] pending_slot;
    logic pending_direct_found;
    logic [READ_PTR_W-1:0] pending_direct;

    wire [31:0] request_line =
        {read_addr_queue[map_cursor][31:6], 6'b0};
    wire [SLOT_W-1:0] head_slot = read_slot_queue[read_queue_head];
    wire [31:0] response_addr =
        read_addr_queue[read_queue_head] + response_index;
    wire head_direct = read_direct_queue[read_queue_head];
    wire response_available = read_queue_count != 0 &&
        read_mapped_queue[read_queue_head] &&
        (head_direct ? direct_ready_queue[read_queue_head] :
                       slot_valid[head_slot]);
    wire byte_handshake = read_data_valid && read_data_ready;
    wire word_handshake = read_word_valid && read_word_ready;
    wire response_finishing =
        (byte_handshake && read_data_last) ||
        (word_handshake && read_word_last);
    wire accept_read = read_req_valid && read_req_ready;
    wire map_direct_eligible = read_mode_queue[map_cursor] &&
                               read_len_queue[map_cursor] == 4 &&
                               read_addr_queue[map_cursor][1:0] == 0;
    wire map_direct = map_direct_eligible && !request_slot_hit;
    wire map_decision_ready = 1'b1;
    wire map_read = unmapped_count != 0 &&
                    map_decision_ready &&
                    (map_direct || request_slot_found);
    wire issue_direct = !write_flush_pending && pending_direct_found;
    wire issue_line = !write_flush_pending && !pending_direct_found &&
                      pending_slot_found;
    wire issue_read = issue_direct || issue_line;
    wire issue_write = write_flush_pending;
    wire accept_mem_cmd = mem_cmd_valid && mem_cmd_ready;
    wire accept_read_cmd = accept_mem_cmd && issue_read;
    wire finish_return = mem_read_valid && mem_read_ready && mem_read_last;
    wire return_is_direct = return_direct_fifo[return_head];
    wire [RETURN_ID_W-1:0] return_id = return_id_fifo[return_head];

    function automatic logic [7:0] select_byte(
        input logic [31:0] word, input logic [1:0] lane
    );
        select_byte = word[8*lane +: 8];
    endfunction

    always_comb begin
        request_slot_found = 1'b0;
        request_slot_hit = 1'b0;
        request_slot = '0;
        // A filling line is a hit: later requests share its one DDR burst.
        for (int slot = 0; slot < CACHE_LINES; slot++) begin
            if (!request_slot_found && slot_allocated[slot] &&
                slot_base[slot] == request_line) begin
                request_slot_found = 1'b1;
                request_slot_hit = 1'b1;
                request_slot = SLOT_W'(slot);
            end
        end
        // Reuse only slots that no accepted request still references.
        for (int slot = 0; slot < CACHE_LINES; slot++) begin
            if (!request_slot_found &&
                (!slot_allocated[slot] ||
                 (slot_refs[slot] == 0 && !slot_cmd_pending[slot] &&
                  !slot_fill_active[slot]))) begin
                request_slot_found = 1'b1;
                request_slot = SLOT_W'(slot);
            end
        end
    end

    always_comb begin
        pending_slot_found = 1'b0;
        pending_slot = '0;
        for (int slot = 0; slot < CACHE_LINES; slot++) begin
            if (!pending_slot_found && slot_cmd_pending[slot]) begin
                pending_slot_found = 1'b1;
                pending_slot = SLOT_W'(slot);
            end
        end
    end

    always_comb begin
        pending_direct_found = 1'b0;
        pending_direct = '0;
        for (int request = 0; request < READ_QUEUE_DEPTH; request++) begin
            if (!pending_direct_found &&
                direct_pending_queue[request]) begin
                pending_direct_found = 1'b1;
                pending_direct = READ_PTR_W'(request);
            end
        end
    end

    assign read_req_ready =
        !error && read_queue_count < READ_QUEUE_DEPTH &&
        write_count == 0 && !write_flush_pending &&
        !write_send_active && !write_inflight;
    assign write_req_ready =
        !error && read_queue_count == 0 &&
        unmapped_count == 0 && return_count == 0 &&
        slot_cmd_pending == 0 && !pending_direct_found &&
        !write_flush_pending && !write_send_active && !write_inflight &&
        (write_count == 0 ||
         write_req_addr == write_base + (write_count << 2));

    assign mem_cmd_valid = issue_write || issue_read;
    assign mem_cmd_write = issue_write;
    assign mem_cmd_addr =
        issue_write ? write_base :
        (issue_direct ? read_addr_queue[pending_direct] :
                        slot_base[pending_slot]);
    assign mem_cmd_len =
        issue_direct ? 4'd0 :
        (issue_line ? BURST_WORDS - 1 :
        (issue_write ? write_burst_len - 1'b1 : 4'd0));
    assign mem_cmd_wdata = write_cache[0];
    assign mem_cmd_wstrb = write_strobe_cache[0];

    assign mem_write_valid = write_send_active;
    assign mem_write_data = write_cache[write_send_index];
    assign mem_write_strb = write_strobe_cache[write_send_index];
    assign mem_write_last =
        write_send_active && write_send_index == write_burst_len - 1'b1;

    assign mem_read_ready = return_count != 0;
    assign read_data_valid =
        response_available && !read_mode_queue[read_queue_head];
    assign read_data = select_byte(
        cache[head_slot][response_addr[5:2]], response_addr[1:0]);
    assign read_data_last =
        read_data_valid &&
        response_index + 1'b1 >= read_len_queue[read_queue_head];
    assign read_word_valid =
        response_available && read_mode_queue[read_queue_head];
    assign read_word_last =
        read_word_valid &&
        response_index + 5'd4 >= read_len_queue[read_queue_head];

    always_comb begin
        read_word_data = '0;
        read_word_keep = '0;
        if (head_direct) begin
            read_word_data = direct_data_queue[read_queue_head];
            read_word_keep = 4'hf;
        end else begin
            for (int lane = 0; lane < 4; lane++) begin
                if (response_index + lane <
                    read_len_queue[read_queue_head]) begin
                    read_word_data[lane*8 +: 8] = select_byte(
                        cache[head_slot][(response_addr[5:0] + lane) >> 2],
                        (response_addr[1:0] + lane) & 2'b11);
                    read_word_keep[lane] = 1'b1;
                end
            end
        end
    end

    assign busy =
        read_queue_count != 0 || return_count != 0 ||
        slot_cmd_pending != 0 || pending_direct_found ||
        write_count != 0 ||
        write_flush_pending || write_send_active ||
        write_inflight || mem_cmd_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            slot_allocated <= '0;
            slot_valid <= '0;
            slot_cmd_pending <= '0;
            slot_fill_active <= '0;
            read_queue_head <= '0;
            read_queue_tail <= '0;
            map_cursor <= '0;
            read_queue_count <= '0;
            unmapped_count <= '0;
            response_index <= '0;
            map_probe_wait <= 1'b0;
            last_map_direct <= 1'b0;
            return_head <= '0;
            return_tail <= '0;
            return_count <= '0;
            fill_word_index <= '0;
            write_flush_pending <= 1'b0;
            write_send_active <= 1'b0;
            write_inflight <= 1'b0;
            write_base <= '0;
            write_count <= '0;
            write_burst_len <= '0;
            write_send_index <= '0;
            error <= 1'b0;
            for (int slot = 0; slot < CACHE_LINES; slot++)
                slot_refs[slot] <= '0;
            for (int request = 0;
                 request < READ_QUEUE_DEPTH; request++) begin
                read_direct_queue[request] <= 1'b0;
                direct_pending_queue[request] <= 1'b0;
                direct_ready_queue[request] <= 1'b0;
                direct_data_queue[request] <= '0;
            end
        end else begin
            // PS/CMA writes bypass this PL-side cache. Invalidate tags at every
            // command boundary so a reused physical address is fetched again.
            if (invalidate_cache) begin
                if (busy)
                    error <= 1'b1;
                else begin
                    slot_allocated <= '0;
                    slot_valid <= '0;
                end
            end

            if (map_read) begin
                map_probe_wait <= 1'b0;
                last_map_direct <= map_direct;
            end

            if (accept_read) begin
                read_addr_queue[read_queue_tail] <= read_req_addr;
                read_len_queue[read_queue_tail] <= read_req_len;
                read_mode_queue[read_queue_tail] <= read_req_word_mode;
                read_mapped_queue[read_queue_tail] <= 1'b0;
                read_direct_queue[read_queue_tail] <= 1'b0;
                direct_pending_queue[read_queue_tail] <= 1'b0;
                direct_ready_queue[read_queue_tail] <= 1'b0;
                read_queue_tail <= read_queue_tail + 1'b1;
                if (read_req_len == 0 || read_req_len > 16 ||
                    {1'b0, read_req_addr[5:0]} + read_req_len > 64)
                    error <= 1'b1;
            end

            // Slot lookup is intentionally behind the ingress FIFO. Producer
            // address buses therefore cannot extend the top-level ready path.
            if (map_read) begin
                read_mapped_queue[map_cursor] <= 1'b1;
                read_direct_queue[map_cursor] <= map_direct;
                map_cursor <= map_cursor + 1'b1;
                if (map_direct) begin
                    direct_pending_queue[map_cursor] <= 1'b1;
                end else begin
                    read_slot_queue[map_cursor] <= request_slot;
                end
                if (!map_direct && !request_slot_hit) begin
                    slot_allocated[request_slot] <= 1'b1;
                    slot_valid[request_slot] <= 1'b0;
                    slot_cmd_pending[request_slot] <= 1'b1;
                    slot_base[request_slot] <= request_line;
                end
            end

            if (byte_handshake || word_handshake) begin
                if (response_finishing) begin
                    read_queue_head <= read_queue_head + 1'b1;
                    response_index <= '0;
                    if (head_direct)
                        direct_ready_queue[read_queue_head] <= 1'b0;
                end else begin
                    response_index <= response_index +
                        (read_mode_queue[read_queue_head] ? 5'd4 : 5'd1);
                end
            end

            unique case ({accept_read, response_finishing})
                2'b10: read_queue_count <= read_queue_count + 1'b1;
                2'b01: read_queue_count <= read_queue_count - 1'b1;
                default: read_queue_count <= read_queue_count;
            endcase

            unique case ({accept_read, map_read})
                2'b10: unmapped_count <= unmapped_count + 1'b1;
                2'b01: unmapped_count <= unmapped_count - 1'b1;
                default: unmapped_count <= unmapped_count;
            endcase

            for (int slot = 0; slot < CACHE_LINES; slot++) begin
                unique case ({
                    map_read && !map_direct &&
                        request_slot == SLOT_W'(slot),
                    response_finishing && !head_direct &&
                        head_slot == SLOT_W'(slot)
                })
                    2'b10: slot_refs[slot] <= slot_refs[slot] + 1'b1;
                    2'b01: slot_refs[slot] <= slot_refs[slot] - 1'b1;
                    default: slot_refs[slot] <= slot_refs[slot];
                endcase
            end

            if (accept_read_cmd) begin
                return_direct_fifo[return_tail] <= issue_direct;
                return_id_fifo[return_tail] <= issue_direct ?
                    RETURN_ID_W'(pending_direct) :
                    RETURN_ID_W'(pending_slot);
                return_tail <= return_tail + 1'b1;
                if (issue_direct) begin
                    direct_pending_queue[pending_direct] <= 1'b0;
                end else begin
                    slot_cmd_pending[pending_slot] <= 1'b0;
                    slot_fill_active[pending_slot] <= 1'b1;
                end
            end

            if (mem_read_valid && mem_read_ready) begin
                if (return_is_direct) begin
                    direct_data_queue[
                        return_id[READ_PTR_W-1:0]] <= mem_read_data;
                    if (!mem_read_last)
                        error <= 1'b1;
                    else
                        direct_ready_queue[
                            return_id[READ_PTR_W-1:0]] <= 1'b1;
                end else begin
                    cache[return_id[SLOT_W-1:0]][fill_word_index] <=
                        mem_read_data;
                    if (mem_read_last) begin
                        if (fill_word_index != BURST_WORDS-1)
                            error <= 1'b1;
                        slot_fill_active[
                            return_id[SLOT_W-1:0]] <= 1'b0;
                        slot_valid[return_id[SLOT_W-1:0]] <= 1'b1;
                        fill_word_index <= '0;
                    end else if (fill_word_index == BURST_WORDS-1) begin
                        error <= 1'b1;
                    end else begin
                        fill_word_index <= fill_word_index + 1'b1;
                    end
                end
                if (mem_read_last)
                    return_head <= return_head + 1'b1;
            end

            unique case ({accept_read_cmd, finish_return})
                2'b10: return_count <= return_count + 1'b1;
                2'b01: return_count <= return_count - 1'b1;
                default: return_count <= return_count;
            endcase

            if (write_req_valid && write_req_ready) begin
                if (write_req_addr[1:0] != 0) begin
                    error <= 1'b1;
                end else begin
                    if (write_count == 0)
                        write_base <= write_req_addr;
                    write_cache[write_count] <= write_req_data;
                    write_strobe_cache[write_count] <= write_req_strb;
                    if (write_req_last ||
                        write_count == BURST_WORDS-1) begin
                        write_burst_len <= write_count + 1'b1;
                        write_flush_pending <= 1'b1;
                    end else begin
                        write_count <= write_count + 1'b1;
                    end
                    slot_allocated <= '0;
                    slot_valid <= '0;
                end
            end

            if (write_req_valid && !write_req_ready &&
                write_count != 0 && !write_flush_pending &&
                !write_send_active && !write_inflight &&
                read_queue_count == 0 && return_count == 0 &&
                slot_cmd_pending == 0 &&
                !pending_direct_found &&
                write_req_addr !=
                    write_base + (write_count << 2)) begin
                write_burst_len <= write_count;
                write_flush_pending <= 1'b1;
            end

            if (accept_mem_cmd && issue_write) begin
                write_flush_pending <= 1'b0;
                write_send_active <= 1'b1;
                write_send_index <= '0;
                write_inflight <= 1'b1;
            end
            if (mem_write_valid && mem_write_ready) begin
                if (mem_write_last)
                    write_send_active <= 1'b0;
                else
                    write_send_index <= write_send_index + 1'b1;
            end
            if (write_inflight && !write_send_active &&
                !write_flush_pending && mem_cmd_ready) begin
                write_inflight <= 1'b0;
                write_count <= '0;
            end
        end
    end
endmodule
