module memory_io_driver #(
    parameter int BURST_WORDS = 16
) (
    input logic clk, input logic rst_n,
    input logic read_req_valid, output logic read_req_ready, input logic [31:0] read_req_addr,
    output logic read_data_valid, input logic read_data_ready, output logic [7:0] read_data,
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
    logic [31:0] cache [0:BURST_WORDS-1];
    logic cache_valid, fill_active, deliver_pending, read_pending;
    logic write_flush_pending, write_send_active, write_inflight;
    logic [31:0] cache_base, requested_addr, write_base;
    logic [3:0] fill_index, write_count, write_burst_len, write_send_index;
    logic [31:0] write_cache [0:BURST_WORDS-1];
    logic [3:0] write_strobe_cache [0:BURST_WORDS-1];
    logic [7:0] read_response_data;
    logic read_response_valid;

    function automatic logic [7:0] select_byte(input logic [31:0] word, input logic [1:0] lane);
        select_byte = word[8*lane +: 8];
    endfunction

    wire [31:0] requested_line = {read_req_addr[31:6], 6'b0};
    wire [3:0] requested_word = read_req_addr[5:2];
    wire cache_hit = cache_valid && cache_base == requested_line;
    wire issue_write = write_flush_pending;
    wire issue_read = !write_flush_pending && read_pending;

    assign read_req_ready = !error && !read_pending && !fill_active && !deliver_pending && !read_response_valid;
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
    assign read_data_valid = read_response_valid;
    assign read_data = read_response_data;
    assign busy = read_pending || fill_active || deliver_pending || read_response_valid || write_count != 0 || write_flush_pending || write_send_active || write_inflight || mem_cmd_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cache_valid <= 1'b0; cache_base <= '0; fill_active <= 1'b0; fill_index <= '0;
            deliver_pending <= 1'b0; read_pending <= 1'b0; requested_addr <= '0;
            read_response_valid <= 1'b0; read_response_data <= '0;
            write_flush_pending <= 1'b0; write_send_active <= 1'b0; write_inflight <= 1'b0;
            write_base <= '0; write_count <= '0; write_burst_len <= '0; write_send_index <= '0;
            error <= 1'b0;
        end else begin
            if (read_response_valid && read_data_ready) read_response_valid <= 1'b0;
            if (read_req_valid && read_req_ready) begin
                requested_addr <= read_req_addr;
                if (cache_hit) begin
                    read_response_data <= select_byte(cache[requested_word], read_req_addr[1:0]);
                    read_response_valid <= 1'b1;
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
            if (write_inflight && !write_send_active && mem_cmd_ready) begin
                write_inflight <= 1'b0; write_count <= '0;
            end
            if (mem_read_valid) begin
                if (!fill_active) error <= 1'b1;
                else begin
                    cache[fill_index] <= mem_read_data;
                    if (mem_read_last) begin
                        if (fill_index != BURST_WORDS-1) error <= 1'b1;
                        fill_active <= 1'b0; cache_valid <= 1'b1; deliver_pending <= 1'b1;
                    end else if (fill_index == BURST_WORDS-1) error <= 1'b1;
                    else fill_index <= fill_index + 1'b1;
                end
            end
            if (deliver_pending && !read_response_valid) begin
                read_response_data <= select_byte(cache[requested_addr[5:2]], requested_addr[1:0]);
                read_response_valid <= 1'b1; deliver_pending <= 1'b0;
            end
        end
    end
endmodule
