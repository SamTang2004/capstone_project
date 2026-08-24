module memory_io_driver #(
    parameter int BURST_WORDS = 16,
    parameter int READ_BURST_BEATS = 8,
    parameter int READ_CACHE_LINES = 8,
    parameter int READ_REQUEST_DEPTH = 4,
    parameter bit READ_AHEAD_ENABLE = 1'b0,
    parameter bit WIDE_RESPONSE_ENABLE = 1'b0
) (
    input logic clk, input logic rst_n,
    input logic read_req_valid, output logic read_req_ready, input logic [31:0] read_req_addr,
    input logic [6:0] read_req_len,
    input logic read_req_wide,
    input logic read_req_qword,
    input logic read_req_allow_ahead,
    output logic read_data_valid, input logic read_data_ready, output logic [7:0] read_data,
    output logic read_data_last,
    output logic read_word_valid, input logic read_word_ready,
    output logic [31:0] read_word_data, output logic [3:0] read_word_keep,
    output logic read_word_last,
    output logic read_qword_valid, input logic read_qword_ready,
    output logic [63:0] read_qword_data, output logic [7:0] read_qword_keep,
    output logic read_qword_last,
    input logic write_req_valid, output logic write_req_ready, input logic [31:0] write_req_addr,
    input logic [31:0] write_req_data, input logic [3:0] write_req_strb, input logic write_req_last,
    output logic mem_cmd_valid, input logic mem_cmd_ready, output logic mem_cmd_write,
    output logic [31:0] mem_cmd_addr, output logic [3:0] mem_cmd_len,
    output logic [31:0] mem_cmd_wdata, output logic [3:0] mem_cmd_wstrb,
    output logic mem_write_valid, input logic mem_write_ready,
    output logic [31:0] mem_write_data, output logic [3:0] mem_write_strb, output logic mem_write_last,
    input logic mem_read_valid, output logic mem_read_ready, input logic [63:0] mem_read_data,
    input logic mem_read_last, output logic busy, output logic error,
    output logic [2:0] error_code, output logic [7:0] debug_first_byte,
    output logic [4:0] debug_fill_beats,
    input logic cache_invalidate,
    input logic stats_reset,
    output logic [31:0] stats_cache_hits, output logic [31:0] stats_cache_misses,
    output logic [31:0] stats_fill_cycles, output logic [31:0] stats_bytes_delivered,
    output logic [31:0] debug_state
);
    // A read request streams 1..64 consecutive bytes from one 64-byte line.
    // Line misses use eight 64-bit AXI beats; legacy consumers may still
    // receive one byte or one 32-bit word at a time.
    localparam int CACHE_INDEX_WIDTH = READ_CACHE_LINES <= 2 ? 1 : $clog2(READ_CACHE_LINES);
    localparam int REQUEST_PTR_WIDTH = READ_REQUEST_DEPTH <= 2 ? 1 : $clog2(READ_REQUEST_DEPTH);
    localparam int REQUEST_COUNT_WIDTH = $clog2(READ_REQUEST_DEPTH + 1);
    logic [63:0] cache [0:READ_CACHE_LINES-1][0:READ_BURST_BEATS-1];
    logic [31:0] cache_base [0:READ_CACHE_LINES-1];
    logic [READ_CACHE_LINES-1:0] cache_valid;
    logic fill_active, deliver_active, read_pending;
    logic fill_is_prefetch;
    logic prefetch_valid, prefetch_filled, prefetch_wide, prefetch_qword;
    logic [31:0] fill_addr, prefetch_addr;
    logic [6:0] prefetch_len;
    logic [CACHE_INDEX_WIDTH-1:0] prefetch_cache_line;
    logic lookup_pending, lookup_result_pending;
    logic prepared_valid, prepared_hit;
    logic write_flush_pending, write_send_active, write_inflight;
    logic [31:0] requested_addr, write_base, write_next_addr;
    logic [6:0] requested_len, response_index;
    logic requested_wide, requested_qword, lookup_wide, lookup_qword, prepared_wide, prepared_qword;
    logic [31:0] lookup_addr, prepared_addr;
    logic [6:0] lookup_len, prepared_len;
    logic read_data_valid_r, read_data_last_r;
    logic [7:0] read_data_r;
    logic [31:0] read_word_data_r;
    logic [3:0] read_word_keep_r;
    logic [31:0] read_word_data_next;
    logic [3:0] read_word_keep_next;
    logic [63:0] read_qword_data_r, read_qword_data_next;
    logic [7:0] read_qword_keep_r, read_qword_keep_next;
    logic [4:0] fill_index;
    logic [CACHE_INDEX_WIDTH-1:0] active_cache_line, fill_cache_line, replace_line;
    logic cache_hit;
    logic [CACHE_INDEX_WIDTH-1:0] cache_hit_line, cache_victim_line;
    logic lookup_hit_r;
    logic [CACHE_INDEX_WIDTH-1:0] lookup_hit_line_r, lookup_victim_line_r;
    logic [CACHE_INDEX_WIDTH-1:0] prepared_hit_line, prepared_victim_line;
    logic [31:0] request_addr_fifo [0:READ_REQUEST_DEPTH-1];
    logic [6:0] request_len_fifo [0:READ_REQUEST_DEPTH-1];
    logic request_wide_fifo [0:READ_REQUEST_DEPTH-1];
    logic request_qword_fifo [0:READ_REQUEST_DEPTH-1];
    logic [REQUEST_PTR_WIDTH-1:0] request_write_ptr, request_read_ptr;
    logic [REQUEST_COUNT_WIDTH-1:0] request_count;
    logic [3:0] write_count, write_burst_len, write_send_index;
    logic [31:0] write_cache [0:BURST_WORDS-1];
    logic [3:0] write_strobe_cache [0:BURST_WORDS-1];
    function automatic logic [7:0] select_byte(input logic [63:0] word, input logic [2:0] lane);
        select_byte = word[8*lane +: 8];
    endfunction

    wire [31:0] lookup_line = {lookup_addr[31:6], 6'b0};
    wire [31:0] response_addr = requested_addr + response_index;
    wire [6:0] response_remaining = requested_len - response_index;
    wire [3:0] response_advance = requested_qword
        ? (response_remaining >= 8 ? 4'd8 : response_remaining[3:0])
        : requested_wide
            ? (response_remaining >= 4 ? 4'd4 : response_remaining[3:0])
            : 4'd1;
    wire response_ready = requested_qword ? read_qword_ready :
                          (requested_wide ? read_word_ready : read_data_ready);
    wire issue_write = write_flush_pending;
    wire issue_read = !write_flush_pending && read_pending;
    wire request_shape_valid = read_req_len != 0 && read_req_len <= 64 &&
                               {1'b0, read_req_addr[5:0]} + read_req_len <= 64;
    wire request_push = read_req_valid && read_req_ready && request_shape_valid;
    wire write_path_busy = write_count != 0 || write_flush_pending || write_send_active || write_inflight;
    wire lookup_can_start = request_count != 0 && !lookup_pending && !lookup_result_pending &&
                            !prepared_valid && !prefetch_valid && !fill_active && !read_pending && !write_path_busy;
    wire request_pop = lookup_can_start;
    wire delivery_finishing = deliver_active && read_data_valid_r && response_ready && read_data_last_r;
    wire response_slot_available = !prefetch_valid && !fill_active && !read_pending &&
                                   (!deliver_active || delivery_finishing);
    wire prefetch_can_start = prepared_valid && !prepared_hit && deliver_active && !delivery_finishing &&
                              !prefetch_valid && !fill_active && !read_pending && !write_path_busy;
    wire prefetch_response_available = prefetch_valid && prefetch_filled &&
                                       (!deliver_active || delivery_finishing);
    wire request_ahead_allowed = READ_AHEAD_ENABLE && read_req_allow_ahead;

    // Legacy consumers permit only one logical request in flight. Request-ahead
    // is opt-in until a consumer can distinguish every queued response.
    // A fused writer and the next tile loader may assert valid together on an
    // idle cycle.  Give the writer admission priority so a read cannot enter
    // the FIFO in the same edge that creates a new partial write burst.
    assign read_req_ready = !error && !read_data_valid_r && !write_req_valid && !write_path_busy &&
                            request_count < READ_REQUEST_DEPTH &&
                            (request_ahead_allowed ||
                             (request_count == 0 && !lookup_pending && !lookup_result_pending && !prepared_valid &&
                              !fill_active && !read_pending && !deliver_active));
    assign write_req_ready = !error && !write_flush_pending && !write_send_active && !write_inflight &&
                             request_count == 0 && !lookup_pending && !lookup_result_pending && !prepared_valid &&
                             !prefetch_valid && !fill_active && !read_pending && !deliver_active &&
                             (write_count == 0 || write_req_addr == write_next_addr);
    assign mem_cmd_valid = issue_write || issue_read;
    assign mem_cmd_write = issue_write;
    assign mem_cmd_addr = issue_write ? write_base : {fill_addr[31:6], 6'b0};
    assign mem_cmd_len = issue_read ? READ_BURST_BEATS - 1 : (issue_write ? write_burst_len - 1'b1 : 4'd0);
    assign mem_cmd_wdata = write_cache[0];
    assign mem_cmd_wstrb = write_strobe_cache[0];
    assign mem_write_valid = write_send_active;
    assign mem_write_data = write_cache[write_send_index];
    assign mem_write_strb = write_strobe_cache[write_send_index];
    assign mem_write_last = write_send_active && write_send_index == write_burst_len - 1'b1;
    assign mem_read_ready = fill_active;
    assign read_data_valid = read_data_valid_r && !requested_wide && !requested_qword;
    assign read_data = read_data_r;
    assign read_data_last = read_data_last_r;
    assign read_word_valid = read_data_valid_r && requested_wide && !requested_qword;
    assign read_word_data = read_word_data_r;
    assign read_word_keep = read_word_keep_r;
    assign read_word_last = read_data_last_r;
    assign read_qword_valid = read_data_valid_r && requested_qword;
    assign read_qword_data = read_qword_data_r;
    assign read_qword_keep = read_qword_keep_r;
    assign read_qword_last = read_data_last_r;
    assign busy = request_count != 0 || lookup_pending || lookup_result_pending || prepared_valid || prefetch_valid ||
                  read_pending || fill_active || deliver_active ||
                  write_count != 0 || write_flush_pending || write_send_active || write_inflight || mem_cmd_valid;
    assign debug_state = {
        read_qword_ready, read_qword_valid, write_req_ready, write_req_valid,
        mem_cmd_ready, mem_cmd_valid, write_inflight, write_send_active,
        write_flush_pending, write_count, read_data_valid_r, deliver_active,
        read_pending, fill_active, prefetch_valid, prepared_valid,
        lookup_result_pending, lookup_pending, request_count
    };

    always_comb begin
        cache_hit = 1'b0;
        cache_hit_line = '0;
        cache_victim_line = replace_line;
        for (int line = 0; line < READ_CACHE_LINES; line++) begin
            if (!cache_valid[line]) cache_victim_line = line[CACHE_INDEX_WIDTH-1:0];
            if (cache_valid[line] && cache_base[line] == lookup_line) begin
                cache_hit = 1'b1;
                cache_hit_line = line[CACHE_INDEX_WIDTH-1:0];
            end
        end
        if (deliver_active && cache_victim_line == active_cache_line)
            cache_victim_line = active_cache_line == READ_CACHE_LINES-1 ? '0 : active_cache_line + 1'b1;
    end

    always_comb begin
        logic [31:0] lane_addr;
        read_word_data_next = '0;
        read_word_keep_next = '0;
        for (int lane = 0; lane < 4; lane++) begin
            lane_addr = response_addr + lane;
            if (!requested_qword && lane < response_advance) begin
                read_word_data_next[lane*8 +: 8] = select_byte(
                    cache[active_cache_line][lane_addr[5:3]], lane_addr[2:0]);
                read_word_keep_next[lane] = 1'b1;
            end
        end
        read_qword_data_next = '0;
        read_qword_keep_next = '0;
        for (int lane = 0; lane < 8; lane++) begin
            lane_addr = response_addr + lane;
            if (requested_qword && lane < response_advance) begin
                read_qword_data_next[lane*8 +: 8] = select_byte(
                    cache[active_cache_line][lane_addr[5:3]], lane_addr[2:0]);
                read_qword_keep_next[lane] = 1'b1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cache_valid <= '0; fill_active <= 1'b0; fill_index <= '0;
            fill_is_prefetch <= 1'b0; fill_addr <= '0;
            prefetch_valid <= 1'b0; prefetch_filled <= 1'b0; prefetch_wide <= 1'b0; prefetch_qword <= 1'b0;
            prefetch_addr <= '0; prefetch_len <= '0; prefetch_cache_line <= '0;
            active_cache_line <= '0; fill_cache_line <= '0; replace_line <= '0;
            deliver_active <= 1'b0; read_pending <= 1'b0; requested_addr <= '0;
            lookup_pending <= 1'b0; lookup_result_pending <= 1'b0;
            prepared_valid <= 1'b0; prepared_hit <= 1'b0;
            lookup_hit_r <= 1'b0; lookup_hit_line_r <= '0; lookup_victim_line_r <= '0;
            lookup_addr <= '0; lookup_len <= '0; lookup_wide <= 1'b0; lookup_qword <= 1'b0;
            prepared_addr <= '0; prepared_len <= '0; prepared_wide <= 1'b0; prepared_qword <= 1'b0;
            prepared_hit_line <= '0; prepared_victim_line <= '0;
            request_write_ptr <= '0; request_read_ptr <= '0; request_count <= '0;
            requested_len <= 7'd1; requested_wide <= 1'b0; requested_qword <= 1'b0; response_index <= '0;
            read_data_valid_r <= 1'b0; read_data_last_r <= 1'b0; read_data_r <= '0;
            read_word_data_r <= '0; read_word_keep_r <= '0;
            read_qword_data_r <= '0; read_qword_keep_r <= '0;
            write_flush_pending <= 1'b0; write_send_active <= 1'b0; write_inflight <= 1'b0;
            write_base <= '0; write_next_addr <= '0;
            write_count <= '0; write_burst_len <= '0; write_send_index <= '0;
            error <= 1'b0; error_code <= 3'd0;
            debug_first_byte <= '0; debug_fill_beats <= '0;
            stats_cache_hits <= '0; stats_cache_misses <= '0;
            stats_fill_cycles <= '0; stats_bytes_delivered <= '0;
        end else begin
            // PS software may update DDR between accelerator dispatches without
            // traversing this driver's write path. Invalidate at that boundary,
            // then retain lines for reuse throughout the operation.
            if (cache_invalidate)
                cache_valid <= '0;
            if (stats_reset) begin
                stats_cache_hits <= '0; stats_cache_misses <= '0;
                stats_fill_cycles <= '0; stats_bytes_delivered <= '0;
            end
            // Register cache data and its metadata as one stream stage. The
            // stage refills in the same cycle that the consumer accepts the
            // previous byte, sustaining one byte per clock without exposing
            // a changing LUTRAM/mux address directly to the consumer.
            if (deliver_active && (!read_data_valid_r || response_ready)) begin
                if (response_index < requested_len) begin
                    read_data_r <= select_byte(cache[active_cache_line][response_addr[5:3]], response_addr[2:0]);
                    read_word_data_r <= read_word_data_next;
                    read_word_keep_r <= read_word_keep_next;
                    read_qword_data_r <= read_qword_data_next;
                    read_qword_keep_r <= read_qword_keep_next;
                    read_data_last_r <= response_index + response_advance >= requested_len;
                    read_data_valid_r <= 1'b1;
                    response_index <= response_index + response_advance;
                end else begin
                    read_data_valid_r <= 1'b0;
                    read_data_last_r <= 1'b0;
                    deliver_active <= 1'b0;
                end
            end
            if (read_req_valid && read_req_ready) begin
                if (!request_shape_valid) begin
                    error <= 1'b1;
                    error_code <= 3'd1;
                end else begin
                    request_addr_fifo[request_write_ptr] <= read_req_addr;
                    request_len_fifo[request_write_ptr] <= read_req_len;
                    request_wide_fifo[request_write_ptr] <= WIDE_RESPONSE_ENABLE && read_req_wide;
                    request_qword_fifo[request_write_ptr] <= WIDE_RESPONSE_ENABLE && read_req_qword;
                    request_write_ptr <= request_write_ptr == READ_REQUEST_DEPTH-1 ? '0 : request_write_ptr + 1'b1;
                end
            end
            if (request_pop) begin
                lookup_addr <= request_addr_fifo[request_read_ptr];
                lookup_len <= request_len_fifo[request_read_ptr];
                lookup_wide <= request_wide_fifo[request_read_ptr];
                lookup_qword <= request_qword_fifo[request_read_ptr];
                request_read_ptr <= request_read_ptr == READ_REQUEST_DEPTH-1 ? '0 : request_read_ptr + 1'b1;
                lookup_pending <= 1'b1;
            end
            unique case ({request_push, request_pop})
                2'b10: request_count <= request_count + 1'b1;
                2'b01: request_count <= request_count - 1'b1;
                default: begin end
            endcase
            // Keep producer address arithmetic and the top-level owner mux
            // outside the associative cache-lookup cone. The cache metadata
            // cannot change while these stages are occupied because both read
            // and write admission are held off.
            if (lookup_pending) begin
                lookup_pending <= 1'b0;
                lookup_result_pending <= 1'b1;
                lookup_hit_r <= cache_hit;
                lookup_hit_line_r <= cache_hit_line;
                lookup_victim_line_r <= cache_victim_line;
            end
            if (lookup_result_pending) begin
                lookup_result_pending <= 1'b0;
                prepared_valid <= 1'b1;
                prepared_hit <= lookup_hit_r;
                prepared_addr <= lookup_addr;
                prepared_len <= lookup_len;
                prepared_wide <= lookup_wide;
                prepared_qword <= lookup_qword;
                prepared_hit_line <= lookup_hit_line_r;
                prepared_victim_line <= lookup_victim_line_r;
            end
            // A completed lookup waits here while the prior response streams.
            // On its final byte, hand the response slot directly to the next
            // request so cache lookup latency is hidden behind useful output.
            if (prepared_valid && response_slot_available) begin
                prepared_valid <= 1'b0;
                requested_addr <= prepared_addr;
                requested_len <= prepared_len;
                requested_wide <= prepared_wide;
                requested_qword <= prepared_qword;
                response_index <= '0;
                read_data_valid_r <= 1'b0;
                read_data_last_r <= 1'b0;
                if (prepared_hit) begin
                    active_cache_line <= prepared_hit_line;
                    stats_cache_hits <= stats_cache_hits + 1'b1;
                    deliver_active <= 1'b1;
                end else begin
                    fill_cache_line <= prepared_victim_line;
                    fill_addr <= prepared_addr;
                    fill_is_prefetch <= 1'b0;
                    stats_cache_misses <= stats_cache_misses + 1'b1;
                    read_pending <= 1'b1;
                end
            end
            // A miss behind an active response may fill a different cache line
            // immediately. Its metadata waits in one ordered prefetch slot.
            if (prefetch_can_start) begin
                prepared_valid <= 1'b0;
                prefetch_valid <= 1'b1;
                prefetch_filled <= 1'b0;
                prefetch_addr <= prepared_addr;
                prefetch_len <= prepared_len;
                prefetch_wide <= prepared_wide;
                prefetch_qword <= prepared_qword;
                prefetch_cache_line <= prepared_victim_line;
                fill_cache_line <= prepared_victim_line;
                fill_addr <= prepared_addr;
                fill_is_prefetch <= 1'b1;
                stats_cache_misses <= stats_cache_misses + 1'b1;
                read_pending <= 1'b1;
            end
            if (prefetch_response_available) begin
                prefetch_valid <= 1'b0;
                prefetch_filled <= 1'b0;
                requested_addr <= prefetch_addr;
                requested_len <= prefetch_len;
                requested_wide <= prefetch_wide;
                requested_qword <= prefetch_qword;
                active_cache_line <= prefetch_cache_line;
                response_index <= '0;
                read_data_valid_r <= 1'b0;
                read_data_last_r <= 1'b0;
                deliver_active <= 1'b1;
            end
            if (write_req_valid && write_req_ready) begin
                if (write_req_addr[1:0] != 0) begin
                    error <= 1'b1;
                    error_code <= 3'd2;
                end
                else begin
                    if (write_count == 0) write_base <= write_req_addr;
                    write_next_addr <= write_req_addr + 32'd4;
                    write_cache[write_count] <= write_req_data;
                    write_strobe_cache[write_count] <= write_req_strb;
                    if (write_req_last || write_count == BURST_WORDS-1) begin
                        write_burst_len <= write_count + 1'b1;
                        write_flush_pending <= 1'b1;
                    end else write_count <= write_count + 1'b1;
                    // Preserve unrelated activation and weight lines across C
                    // writeback. Only a line actually touched by this word is
                    // invalidated.
                    for (int line = 0; line < READ_CACHE_LINES; line++) begin
                        if (cache_valid[line] &&
                            cache_base[line] == {write_req_addr[31:6], 6'b0})
                            cache_valid[line] <= 1'b0;
                    end
                end
            end
            // A producer may move to a non-contiguous address (for example,
            // after completing one NCHW channel). Drain the accumulated run
            // before accepting that next request instead of holding ready low
            // forever waiting for a continuation that cannot arrive.
            if (write_req_valid && !write_req_ready && write_count != 0 &&
                !write_flush_pending && !write_send_active && !write_inflight &&
                request_count == 0 && !lookup_pending && !lookup_result_pending && !prepared_valid &&
                !fill_active && !read_pending && !deliver_active &&
                write_req_addr != write_next_addr) begin
                write_burst_len <= write_count;
                write_flush_pending <= 1'b1;
            end
            // A read arriving behind a short, still-coalescing write run must
            // force that partial run to DDR before it can enter the request
            // FIFO.  Otherwise the queued read prevents further writer beats
            // while write_count prevents read lookup: a circular admission
            // dependency exposed by GEMM writeback/load overlap.
            if (read_req_valid && !read_req_ready && !write_req_valid &&
                write_count != 0 &&
                !write_flush_pending && !write_send_active && !write_inflight &&
                request_count == 0 && !lookup_pending && !lookup_result_pending &&
                !prepared_valid && !prefetch_valid && !fill_active &&
                !read_pending && !deliver_active) begin
                write_burst_len <= write_count;
                write_flush_pending <= 1'b1;
            end
            if (mem_cmd_valid && mem_cmd_ready) begin
                if (issue_write) begin
                    write_flush_pending <= 1'b0; write_send_active <= 1'b1; write_send_index <= '0; write_inflight <= 1'b1;
                end
                else begin
                    read_pending <= 1'b0; fill_active <= 1'b1; fill_index <= '0;
                    cache_base[fill_cache_line] <= {fill_addr[31:6], 6'b0};
                    cache_valid[fill_cache_line] <= 1'b0;
                end
            end
            if (mem_write_valid && mem_write_ready) begin
                if (mem_write_last) write_send_active <= 1'b0;
                else write_send_index <= write_send_index + 1'b1;
            end
            // mem_cmd_ready indicates that the bridge is idle.  When a new
            // queued burst is accepted in that same cycle, do not clear the
            // in-flight marker for the newly launched transaction.  Losing it
            // permits the next producer write to overtake AXI B completion.
            if (write_inflight && !write_send_active && !write_flush_pending && mem_cmd_ready) begin
                write_inflight <= 1'b0; write_count <= '0;
            end
            // AXI VALID may remain asserted while READY is low. Only a real
            // transfer advances the line fill; backpressure is not an error.
            if (mem_read_valid && mem_read_ready) begin
                stats_fill_cycles <= stats_fill_cycles + 1'b1;
                if (fill_index < READ_BURST_BEATS) cache[fill_cache_line][fill_index[2:0]] <= mem_read_data;
                if (fill_index == 0) debug_first_byte <= mem_read_data[7:0];
                debug_fill_beats <= fill_index + 1'b1;
                if (mem_read_last) begin
                    fill_active <= 1'b0;
                    if (fill_index != READ_BURST_BEATS-1) begin
                        error <= 1'b1;
                        error_code <= 3'd3;
                        cache_valid[fill_cache_line] <= 1'b0;
                    end else begin
                        cache_valid[fill_cache_line] <= 1'b1;
                        if (fill_is_prefetch) begin
                            prefetch_filled <= 1'b1;
                        end else begin
                            active_cache_line <= fill_cache_line;
                            deliver_active <= 1'b1;
                        end
                        replace_line <= fill_cache_line == READ_CACHE_LINES-1 ? '0 : fill_cache_line + 1'b1;
                    end
                end else if (fill_index == READ_BURST_BEATS-1) begin
                    // Keep READY asserted to drain a malformed physical burst,
                    // but never publish the overlong line to consumers.
                    error <= 1'b1;
                    error_code <= 3'd4;
                    cache_valid[fill_cache_line] <= 1'b0;
                    fill_index <= fill_index + 1'b1;
                end else if (fill_index == 5'd31) begin
                    error <= 1'b1;
                    error_code <= 3'd4;
                end
                else fill_index <= fill_index + 1'b1;
            end
            if (read_qword_valid && read_qword_ready) begin
                stats_bytes_delivered <= stats_bytes_delivered +
                    read_qword_keep[0] + read_qword_keep[1] + read_qword_keep[2] + read_qword_keep[3] +
                    read_qword_keep[4] + read_qword_keep[5] + read_qword_keep[6] + read_qword_keep[7];
            end else if (read_word_valid && read_word_ready) begin
                stats_bytes_delivered <= stats_bytes_delivered + read_word_keep[0] +
                    read_word_keep[1] + read_word_keep[2] + read_word_keep[3];
            end else if (read_data_valid && read_data_ready) begin
                stats_bytes_delivered <= stats_bytes_delivered + 1'b1;
            end
        end
    end
endmodule
