module tb_memory_io_load_balance_stress #(
    parameter int REQUEST_BYTES = 1
);
    localparam int TOTAL_BYTES = 1024;
    localparam int REQUESTS = TOTAL_BYTES / REQUEST_BYTES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [6:0] read_req_len;
    logic read_data_valid, read_data_ready;
    logic [7:0] read_data;
    logic read_data_last;
    logic read_word_valid, read_word_ready, read_word_last;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic read_qword_valid, read_qword_ready, read_qword_last;
    logic [63:0] read_qword_data;
    logic [7:0] read_qword_keep;
    logic write_req_valid, write_req_ready, write_req_last;
    logic [31:0] write_req_addr, write_req_data;
    logic [3:0] write_req_strb;
    logic mem_cmd_valid, mem_cmd_ready, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_ready, mem_write_last;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_read_valid, mem_read_ready, mem_read_last;
    logic [63:0] mem_read_data;
    logic busy, error;
    logic [2:0] error_code;
    logic [7:0] debug_first_byte;
    logic [4:0] debug_fill_beats;
    logic cache_invalidate, stats_reset;
    logic [31:0] stats_cache_hits, stats_cache_misses;
    logic [31:0] stats_fill_cycles, stats_bytes_delivered;

    logic [31:0] memory [0:511];
    logic [31:0] active_base;
    logic [3:0] active_len;
    integer active_beat;
    integer issued, received, cycle_count;
    integer command_count, payload_beats, output_beats;
    integer trace_fd;
    string trace_path;

    always #5 clk = ~clk;

    memory_io_driver #(
        .READ_AHEAD_ENABLE(1'b1),
        .READ_REQUEST_DEPTH(REQUEST_BYTES > 1 ? 8 : 4),
        .WIDE_RESPONSE_ENABLE(REQUEST_BYTES > 1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .read_req_valid(read_req_valid), .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr), .read_req_len(read_req_len),
        .read_req_wide(REQUEST_BYTES > 1),
        .read_req_qword(REQUEST_BYTES > 1),
        .read_req_allow_ahead(1'b1),
        .read_data_valid(read_data_valid), .read_data_ready(read_data_ready),
        .read_data(read_data), .read_data_last(read_data_last),
        .read_word_valid(read_word_valid), .read_word_ready(read_word_ready),
        .read_word_data(read_word_data), .read_word_keep(read_word_keep),
        .read_word_last(read_word_last),
        .read_qword_valid(read_qword_valid), .read_qword_ready(read_qword_ready),
        .read_qword_data(read_qword_data), .read_qword_keep(read_qword_keep),
        .read_qword_last(read_qword_last),
        .write_req_valid(write_req_valid), .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr), .write_req_data(write_req_data),
        .write_req_strb(write_req_strb), .write_req_last(write_req_last),
        .mem_cmd_valid(mem_cmd_valid), .mem_cmd_ready(mem_cmd_ready),
        .mem_cmd_write(mem_cmd_write), .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_len(mem_cmd_len), .mem_cmd_wdata(mem_cmd_wdata),
        .mem_cmd_wstrb(mem_cmd_wstrb), .mem_write_valid(mem_write_valid),
        .mem_write_ready(mem_write_ready), .mem_write_data(mem_write_data),
        .mem_write_strb(mem_write_strb), .mem_write_last(mem_write_last),
        .mem_read_valid(mem_read_valid), .mem_read_ready(mem_read_ready),
        .mem_read_data(mem_read_data), .mem_read_last(mem_read_last),
        .busy(busy), .error(error), .error_code(error_code),
        .debug_first_byte(debug_first_byte), .debug_fill_beats(debug_fill_beats),
        .cache_invalidate(cache_invalidate),
        .stats_reset(stats_reset), .stats_cache_hits(stats_cache_hits),
        .stats_cache_misses(stats_cache_misses), .stats_fill_cycles(stats_fill_cycles),
        .stats_bytes_delivered(stats_bytes_delivered)
    );

    assign read_req_valid = rst_n && issued < REQUESTS;
    assign read_req_addr = issued * REQUEST_BYTES;
    assign read_req_len = REQUEST_BYTES;
    assign read_data_ready = REQUEST_BYTES == 1;
    assign read_word_ready = 1'b0;
    assign read_qword_ready = REQUEST_BYTES > 1;
    assign write_req_valid = 1'b0;
    assign write_req_addr = '0;
    assign write_req_data = '0;
    assign write_req_strb = '0;
    assign write_req_last = 1'b0;
    assign mem_cmd_ready = 1'b1;
    assign mem_write_ready = 1'b1;
    assign cache_invalidate = 1'b0;
    assign stats_reset = 1'b0;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            issued <= 0;
            received <= 0;
            cycle_count <= 0;
            command_count <= 0;
            payload_beats <= 0;
            output_beats <= 0;
            mem_read_valid <= 1'b0;
            mem_read_last <= 1'b0;
            mem_read_data <= '0;
            active_base <= '0;
            active_len <= '0;
            active_beat <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (read_req_valid && read_req_ready) issued <= issued + 1;
            if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write) begin
                command_count <= command_count + 1;
                active_base <= mem_cmd_addr;
                active_len <= mem_cmd_len;
                active_beat <= 0;
                mem_read_data <= {memory[(mem_cmd_addr >> 2) + 1], memory[mem_cmd_addr >> 2]};
                mem_read_valid <= 1'b1;
                mem_read_last <= mem_cmd_len == 0;
            end else if (mem_read_valid && mem_read_ready) begin
                payload_beats <= payload_beats + 1;
                if (mem_read_last) begin
                    mem_read_valid <= 1'b0;
                    mem_read_last <= 1'b0;
                end else begin
                    active_beat <= active_beat + 1;
                    mem_read_data <= {
                        memory[(active_base >> 2) + (active_beat + 1) * 2 + 1],
                        memory[(active_base >> 2) + (active_beat + 1) * 2]
                    };
                    mem_read_last <= active_beat + 1 == active_len;
                end
            end
            if (REQUEST_BYTES == 1 && read_data_valid && read_data_ready) begin
                if (read_data !== received[7:0] ||
                    read_data_last !== ((received + 1) % REQUEST_BYTES == 0)) begin
                    $fatal(1, "response mismatch index=%0d data=%02x last=%0b", received, read_data, read_data_last);
                end
                received <= received + 1;
                output_beats <= output_beats + 1;
            end
            if (REQUEST_BYTES > 1 && read_qword_valid && read_qword_ready) begin
                integer valid_bytes;
                valid_bytes = 0;
                for (int lane = 0; lane < 8; lane++) begin
                    if (read_qword_keep[lane]) begin
                        if (read_qword_data[lane*8 +: 8] !== ((received + valid_bytes) & 8'hff))
                            $fatal(1, "wide response mismatch index=%0d lane=%0d data=%02x",
                                received + valid_bytes, lane, read_qword_data[lane*8 +: 8]);
                        valid_bytes++;
                    end
                end
                if (valid_bytes == 0 ||
                    read_qword_last !== ((received + valid_bytes) % REQUEST_BYTES == 0))
                    $fatal(1, "wide response boundary mismatch received=%0d keep=%h last=%0b",
                        received, read_qword_keep, read_qword_last);
                received <= received + valid_bytes;
                output_beats <= output_beats + valid_bytes;
            end

            $fwrite(trace_fd,
                "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                cycle_count,
                read_req_valid, read_req_ready, read_req_valid && read_req_ready,
                mem_cmd_valid, mem_cmd_ready, mem_cmd_valid && mem_cmd_ready && !mem_cmd_write,
                mem_read_valid, mem_read_ready, mem_read_valid && mem_read_ready,
                REQUEST_BYTES > 1 ? read_qword_valid : read_data_valid,
                REQUEST_BYTES > 1 ? read_qword_ready : read_data_ready,
                REQUEST_BYTES > 1 ? read_qword_valid && read_qword_ready : read_data_valid && read_data_ready,
                busy, dut.lookup_pending, dut.lookup_result_pending,
                dut.fill_active, dut.deliver_active, dut.read_pending,
                dut.prepared_valid, dut.request_count,
                REQUEST_BYTES > 1
                    ? ((read_qword_valid && read_qword_ready) ? $countones(read_qword_keep) : 0)
                    : (read_data_valid && read_data_ready),
                dut.prefetch_valid, dut.prefetch_filled);

            if (received == TOTAL_BYTES && !busy) begin
                $display("Memory IO load-balance stress golden trace passed: request_bytes=%0d cycles=%0d requests=%0d commands=%0d payload_beats=%0d output_bytes=%0d hits=%0d misses=%0d",
                    REQUEST_BYTES,
                    cycle_count, issued, command_count, payload_beats, output_beats,
                    stats_cache_hits, stats_cache_misses);
                $fclose(trace_fd);
                $finish;
            end
            if (cycle_count > 50000) $fatal(1, "stress timeout issued=%0d received=%0d busy=%0b", issued, received, busy);
        end
    end

    initial begin
        for (int word = 0; word < 512; word++) begin
            memory[word] = '0;
            for (int lane = 0; lane < 4; lane++) begin
                memory[word][lane*8 +: 8] = (word * 4 + lane) & 8'hff;
            end
        end
        if (!$value$plusargs("TRACE=%s", trace_path)) trace_path = "memory_io_stress.csv";
        trace_fd = $fopen(trace_path, "w");
        if (trace_fd == 0) $fatal(1, "cannot open stress trace %s", trace_path);
        $fwrite(trace_fd,
            "cycle,req_valid,req_ready,req_accept,cmd_valid,cmd_ready,cmd_accept,read_valid,read_ready,read_beat,out_valid,out_ready,out_beat,busy,lookup_pending,lookup_result_pending,fill_active,deliver_active,read_pending,prepared_valid,request_count,out_bytes,prefetch_valid,prefetch_filled\n");
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
    end
endmodule

module tb_memory_io_load_balance_vector_stress;
    tb_memory_io_load_balance_stress #(.REQUEST_BYTES(16)) test();
endmodule

module tb_memory_io_load_balance_line_stress;
    tb_memory_io_load_balance_stress #(.REQUEST_BYTES(64)) test();
endmodule
