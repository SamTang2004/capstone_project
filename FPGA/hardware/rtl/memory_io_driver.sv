module memory_io_driver #(
    parameter int BURST_WORDS = 16
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
    input logic mem_read_last, output logic busy, output logic error
);
    // A read request streams 1..16 consecutive bytes from one 64-byte line.
    // Line misses use the existing 16-word AXI burst; hits stream from cache.
    logic [31:0] cache [0:BURST_WORDS-1];
    logic cache_valid, fill_active, deliver_active, read_pending;
    logic write_flush_pending, write_send_active, write_inflight;
    logic [31:0] cache_base, requested_addr, write_base;
    logic [4:0] requested_len, response_index;
    logic [3:0] fill_index, write_count, write_burst_len, write_send_index;
    logic [31:0] write_cache [0:BURST_WORDS-1];
    logic [3:0] write_strobe_cache [0:BURST_WORDS-1];
    function automatic logic [7:0] select_byte(input logic [31:0] word, input logic [1:0] lane);
        select_byte = word[8*lane +: 8];
    endfunction

    wire [31:0] requested_line = {read_req_addr[31:6], 6'b0};
    wire cache_hit = cache_valid && cache_base == requested_line;
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
    assign read_data_valid = deliver_active;
    assign read_data = select_byte(cache[response_addr[5:2]], response_addr[1:0]);
    assign read_data_last = deliver_active && response_index + 1'b1 >= requested_len;
    assign busy = read_pending || fill_active || deliver_active || write_count != 0 || write_flush_pending || write_send_active || write_inflight || mem_cmd_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cache_valid <= 1'b0; cache_base <= '0; fill_active <= 1'b0; fill_index <= '0;
            deliver_active <= 1'b0; read_pending <= 1'b0; requested_addr <= '0;
            requested_len <= 5'd1; response_index <= '0;
            write_flush_pending <= 1'b0; write_send_active <= 1'b0; write_inflight <= 1'b0;
            write_base <= '0; write_count <= '0; write_burst_len <= '0; write_send_index <= '0;
            error <= 1'b0;
        end else begin
            if (read_data_valid && read_data_ready) begin
                if (read_data_last) deliver_active <= 1'b0;
                else response_index <= response_index + 1'b1;
            end
            if (read_req_valid && read_req_ready) begin
                requested_addr <= read_req_addr;
                requested_len <= read_req_len;
                response_index <= '0;
                if (read_req_len == 0 || read_req_len > 16 || {1'b0, read_req_addr[5:0]} + read_req_len > 64) begin
                    error <= 1'b1;
                end else if (cache_hit) begin
                    deliver_active <= 1'b1;
                end else begin
                    read_pending <= 1'b1;
                end
            end
            if (write_req_valid && write_req_ready) begin
                if (write_req_addr[1:0] != 0) error <= 1'b1;
                else begin
                    if (write_count == 0) write_base <= write_req_addr;
                    write_cache[write_count] <= write_req_data;
                    write_strobe_cache[write_count] <= write_req_strb;
                    if (write_req_last || write_count == BURST_WORDS-1) begin
                        write_burst_len <= write_count + 1'b1;
                        write_flush_pending <= 1'b1;
                    end else write_count <= write_count + 1'b1;
                    cache_valid <= 1'b0;
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
                    cache_base <= {requested_addr[31:6], 6'b0}; cache_valid <= 1'b0;
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
            if (mem_read_valid) begin
                if (!fill_active) error <= 1'b1;
                else begin
                    cache[fill_index] <= mem_read_data;
                    if (mem_read_last) begin
                        if (fill_index != BURST_WORDS-1) error <= 1'b1;
                        fill_active <= 1'b0; cache_valid <= 1'b1; deliver_active <= 1'b1;
                    end else if (fill_index == BURST_WORDS-1) error <= 1'b1;
                    else fill_index <= fill_index + 1'b1;
                end
            end
        end
    end
endmodule
