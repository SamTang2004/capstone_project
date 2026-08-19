module memory_io_driver #(
    parameter int BURST_WORDS = 16,
    parameter int READ_CACHE_LINES = 8
) (
    input logic clk, input logic rst_n,
    input logic read_req_valid, output logic read_req_ready, input logic [31:0] read_req_addr,
    input logic [4:0] read_req_len,
    output logic read_data_valid, input logic read_data_ready, output logic [7:0] read_data,
    output logic read_data_last,
    input logic write_req_valid, output logic write_req_ready, input logic [31:0] write_req_addr,
    input logic [31:0] write_req_data, input logic [3:0] write_req_strb, input logic write_req_last,
    output logic mem_cmd_valid, input logic mem_cmd_ready, output logic mem_cmd_write,
    output logic [31:0] mem_cmd_addr, output logic [3:0] mem_cmd_len,
    output logic [31:0] mem_cmd_wdata, output logic [3:0] mem_cmd_wstrb,
    output logic mem_write_valid, input logic mem_write_ready,
    output logic [31:0] mem_write_data, output logic [3:0] mem_write_strb, output logic mem_write_last,
    input logic mem_read_valid, output logic mem_read_ready, input logic [31:0] mem_read_data,
    input logic mem_read_last, output logic busy, output logic error,
    output logic [2:0] error_code, output logic [7:0] debug_first_byte,
    output logic [4:0] debug_fill_beats,
    input logic stats_reset,
    output logic [31:0] stats_cache_hits, output logic [31:0] stats_cache_misses,
    output logic [31:0] stats_fill_cycles, output logic [31:0] stats_bytes_delivered
);
    // A read request streams 1..16 consecutive bytes from one 64-byte line.
    // Line misses use the existing 16-word AXI burst; hits stream from cache.
    localparam int CACHE_INDEX_WIDTH = READ_CACHE_LINES <= 2 ? 1 : $clog2(READ_CACHE_LINES);
    logic [31:0] cache [0:READ_CACHE_LINES-1][0:BURST_WORDS-1];
    logic [31:0] cache_base [0:READ_CACHE_LINES-1];
    logic [READ_CACHE_LINES-1:0] cache_valid;
    logic fill_active, deliver_active, read_pending;
    logic write_flush_pending, write_send_active, write_inflight;
    logic [31:0] requested_addr, write_base;
    logic [4:0] requested_len, response_index;
    logic read_data_valid_r, read_data_last_r;
    logic [7:0] read_data_r;
    logic [4:0] fill_index;
    logic [CACHE_INDEX_WIDTH-1:0] active_cache_line, fill_cache_line, replace_line;
    logic cache_hit;
    logic [CACHE_INDEX_WIDTH-1:0] cache_hit_line, cache_victim_line;
    logic [3:0] write_count, write_burst_len, write_send_index;
    logic [31:0] write_cache [0:BURST_WORDS-1];
    logic [3:0] write_strobe_cache [0:BURST_WORDS-1];
    function automatic logic [7:0] select_byte(input logic [31:0] word, input logic [1:0] lane);
        select_byte = word[8*lane +: 8];
    endfunction

    wire [31:0] requested_line = {read_req_addr[31:6], 6'b0};
    wire [31:0] response_addr = requested_addr + response_index;
    wire issue_write = write_flush_pending;
    wire issue_read = !write_flush_pending && read_pending;

    assign read_req_ready = !error && !read_pending && !fill_active && !deliver_active;
    assign write_req_ready = !error && !write_flush_pending && !write_send_active && !write_inflight && !fill_active && !read_pending &&
                             (write_count == 0 || write_req_addr == write_base + (write_count << 2));
    assign mem_cmd_valid = issue_write || issue_read;
    assign mem_cmd_write = issue_write;
    assign mem_cmd_addr = issue_write ? write_base : {requested_addr[31:6], 6'b0};
    assign mem_cmd_len = issue_read ? BURST_WORDS - 1 : (issue_write ? write_burst_len - 1'b1 : 4'd0);
    assign mem_cmd_wdata = write_cache[0];
    assign mem_cmd_wstrb = write_strobe_cache[0];
    assign mem_write_valid = write_send_active;
    assign mem_write_data = write_cache[write_send_index];
    assign mem_write_strb = write_strobe_cache[write_send_index];
    assign mem_write_last = write_send_active && write_send_index == write_burst_len - 1'b1;
    assign mem_read_ready = fill_active;
    assign read_data_valid = read_data_valid_r;
    assign read_data = read_data_r;
    assign read_data_last = read_data_last_r;
    assign busy = read_pending || fill_active || deliver_active || write_count != 0 || write_flush_pending || write_send_active || write_inflight || mem_cmd_valid;

    always_comb begin
        cache_hit = 1'b0;
        cache_hit_line = '0;
        cache_victim_line = replace_line;
        for (int line = 0; line < READ_CACHE_LINES; line++) begin
            if (!cache_valid[line]) cache_victim_line = line[CACHE_INDEX_WIDTH-1:0];
            if (cache_valid[line] && cache_base[line] == requested_line) begin
                cache_hit = 1'b1;
                cache_hit_line = line[CACHE_INDEX_WIDTH-1:0];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cache_valid <= '0; fill_active <= 1'b0; fill_index <= '0;
            active_cache_line <= '0; fill_cache_line <= '0; replace_line <= '0;
            deliver_active <= 1'b0; read_pending <= 1'b0; requested_addr <= '0;
            requested_len <= 5'd1; response_index <= '0;
            read_data_valid_r <= 1'b0; read_data_last_r <= 1'b0; read_data_r <= '0;
            write_flush_pending <= 1'b0; write_send_active <= 1'b0; write_inflight <= 1'b0;
            write_base <= '0; write_count <= '0; write_burst_len <= '0; write_send_index <= '0;
            error <= 1'b0; error_code <= 3'd0;
            debug_first_byte <= '0; debug_fill_beats <= '0;
            stats_cache_hits <= '0; stats_cache_misses <= '0;
            stats_fill_cycles <= '0; stats_bytes_delivered <= '0;
        end else begin
            if (stats_reset) begin
                stats_cache_hits <= '0; stats_cache_misses <= '0;
                stats_fill_cycles <= '0; stats_bytes_delivered <= '0;
            end
            // Register cache data and its metadata as one stream stage. The
            // stage refills in the same cycle that the consumer accepts the
            // previous byte, sustaining one byte per clock without exposing
            // a changing LUTRAM/mux address directly to the consumer.
            if (deliver_active && (!read_data_valid_r || read_data_ready)) begin
                if (response_index < requested_len) begin
                    read_data_r <= select_byte(cache[active_cache_line][response_addr[5:2]], response_addr[1:0]);
                    read_data_last_r <= response_index + 1'b1 >= requested_len;
                    read_data_valid_r <= 1'b1;
                    response_index <= response_index + 1'b1;
                end else begin
                    read_data_valid_r <= 1'b0;
                    read_data_last_r <= 1'b0;
                    deliver_active <= 1'b0;
                end
            end
            if (read_req_valid && read_req_ready) begin
                requested_addr <= read_req_addr;
                requested_len <= read_req_len;
                response_index <= '0;
                read_data_valid_r <= 1'b0;
                read_data_last_r <= 1'b0;
                if (read_req_len == 0 || read_req_len > 16 || {1'b0, read_req_addr[5:0]} + read_req_len > 64) begin
                    error <= 1'b1;
                    error_code <= 3'd1;
                end else if (cache_hit) begin
                    active_cache_line <= cache_hit_line;
                    stats_cache_hits <= stats_cache_hits + 1'b1;
                    deliver_active <= 1'b1;
                end else begin
                    fill_cache_line <= cache_victim_line;
                    stats_cache_misses <= stats_cache_misses + 1'b1;
                    read_pending <= 1'b1;
                end
            end
            if (write_req_valid && write_req_ready) begin
                if (write_req_addr[1:0] != 0) begin
                    error <= 1'b1;
                    error_code <= 3'd2;
                end
                else begin
                    if (write_count == 0) write_base <= write_req_addr;
                    write_cache[write_count] <= write_req_data;
                    write_strobe_cache[write_count] <= write_req_strb;
                    if (write_req_last || write_count == BURST_WORDS-1) begin
                        write_burst_len <= write_count + 1'b1;
                        write_flush_pending <= 1'b1;
                    end else write_count <= write_count + 1'b1;
                    cache_valid <= '0;
                end
            end
            // A producer may move to a non-contiguous address (for example,
            // after completing one NCHW channel). Drain the accumulated run
            // before accepting that next request instead of holding ready low
            // forever waiting for a continuation that cannot arrive.
            if (write_req_valid && !write_req_ready && write_count != 0 &&
                !write_flush_pending && !write_send_active && !write_inflight &&
                !fill_active && !read_pending && write_req_addr != write_base + (write_count << 2)) begin
                write_burst_len <= write_count;
                write_flush_pending <= 1'b1;
            end
            if (mem_cmd_valid && mem_cmd_ready) begin
                if (issue_write) begin
                    write_flush_pending <= 1'b0; write_send_active <= 1'b1; write_send_index <= '0; write_inflight <= 1'b1;
                end
                else begin
                    read_pending <= 1'b0; fill_active <= 1'b1; fill_index <= '0;
                    cache_base[fill_cache_line] <= {requested_addr[31:6], 6'b0};
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
                if (fill_index < BURST_WORDS) cache[fill_cache_line][fill_index[3:0]] <= mem_read_data;
                if (fill_index == 0) debug_first_byte <= mem_read_data[7:0];
                debug_fill_beats <= fill_index + 1'b1;
                if (mem_read_last) begin
                    fill_active <= 1'b0;
                    if (fill_index != BURST_WORDS-1) begin
                        error <= 1'b1;
                        error_code <= 3'd3;
                        cache_valid[fill_cache_line] <= 1'b0;
                    end else begin
                        cache_valid[fill_cache_line] <= 1'b1;
                        active_cache_line <= fill_cache_line;
                        replace_line <= fill_cache_line == READ_CACHE_LINES-1 ? '0 : fill_cache_line + 1'b1;
                        deliver_active <= 1'b1;
                    end
                end else if (fill_index == BURST_WORDS-1) begin
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
            if (read_data_valid_r && read_data_ready) begin
                stats_bytes_delivered <= stats_bytes_delivered + 1'b1;
            end
        end
    end
endmodule
