`timescale 1ns/1ps
module tb_memory_io_driver_multiline;
    logic clk = 0, rst_n = 0;
    logic invalidate_cache = 0;
    logic read_req_valid = 0, read_req_ready;
    logic [31:0] read_req_addr = 0;
    logic [4:0] read_req_len = 1;
    logic read_req_word_mode = 0;
    logic read_data_valid, read_data_ready = 1;
    logic [7:0] read_data;
    logic read_data_last;
    logic read_word_valid, read_word_ready = 0;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic read_word_last;
    logic write_req_valid = 0, write_req_ready;
    logic [31:0] write_req_addr = 0, write_req_data = 0;
    logic [3:0] write_req_strb = 0;
    logic write_req_last = 0;
    logic mem_cmd_valid, mem_cmd_ready = 1, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_ready = 1;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_write_last;
    logic mem_read_valid = 0, mem_read_ready;
    logic [31:0] mem_read_data;
    logic mem_read_last = 0;
    logic busy, error;

    logic [31:0] accepted_base [0:3];
    integer command_count = 0;
    integer response_burst = 0;
    integer response_beat = 0;
    integer received = 0;

    always #5 clk = ~clk;

    function automatic logic [31:0] memory_word(
        input logic [31:0] base, input integer beat
    );
        logic [7:0] byte_base;
        begin
            byte_base = base[7:0] + beat * 4;
            memory_word = {
                8'(byte_base + 3), 8'(byte_base + 2),
                8'(byte_base + 1), byte_base
            };
        end
    endfunction

    memory_io_driver #(.CACHE_LINES(4)) dut (.*);

    // Refuse to return any data until all four line commands are visible.
    // A single-line blocking driver deadlocks this memory model.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            command_count <= 0;
            response_burst <= 0;
            response_beat <= 0;
            mem_read_valid <= 0;
            mem_read_last <= 0;
        end else begin
            if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write) begin
                if (mem_cmd_len != 15)
                    $fatal(1, "unexpected burst len=%0d", mem_cmd_len);
                accepted_base[command_count] <= mem_cmd_addr;
                command_count <= command_count + 1;
            end

            if (!mem_read_valid && command_count == 4 &&
                response_burst < 4) begin
                response_beat <= 0;
                mem_read_data <=
                    memory_word(accepted_base[response_burst], 0);
                mem_read_last <= 0;
                mem_read_valid <= 1;
            end else if (mem_read_valid && mem_read_ready) begin
                if (mem_read_last) begin
                    mem_read_valid <= 0;
                    mem_read_last <= 0;
                    response_burst <= response_burst + 1;
                end else begin
                    response_beat <= response_beat + 1;
                    mem_read_data <= memory_word(
                        accepted_base[response_burst],
                        response_beat + 1);
                    mem_read_last <= response_beat == 14;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (read_data_valid && read_data_ready) begin
            if (read_data !== (received * 8'h40 + 8'h03) ||
                !read_data_last)
                $fatal(1, "response[%0d]=%0h last=%0b addr=%h slot=%0d word=%h lane=%0d",
                       received, read_data, read_data_last,
                       dut.read_addr_queue[dut.read_queue_head],
                       dut.head_slot,
                       dut.cache[dut.head_slot][dut.response_addr[5:2]],
                       dut.response_addr[1:0]);
            received <= received + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (int request = 0; request < 4; request++) begin
            @(negedge clk);
            read_req_addr = 32'h100 + request * 32'h40 + 3;
            read_req_valid = 1;
            while (!read_req_ready)
                @(negedge clk);
            @(posedge clk);
        end
        @(negedge clk);
        read_req_valid = 0;
        wait (command_count == 4);
        if (received != 0)
            $fatal(1, "responses arrived before four line commands");
        wait (received == 4);
        if (error)
            $fatal(1, "memory driver asserted error");
        $display("Memory IO multiline golden trace passed: four line misses outstanding");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout commands=%0d bursts=%0d responses=%0d",
               command_count, response_burst, received);
    end
endmodule
