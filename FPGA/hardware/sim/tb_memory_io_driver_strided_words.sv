`timescale 1ns/1ps
module tb_memory_io_driver_strided_words;
    logic clk = 0, rst_n = 0;
    logic invalidate_cache = 0;
    logic read_req_valid = 0, read_req_ready;
    logic [31:0] read_req_addr = 0;
    logic [4:0] read_req_len = 4;
    logic read_req_word_mode = 1;
    logic read_data_valid, read_data_ready = 0;
    logic [7:0] read_data;
    logic read_data_last;
    logic read_word_valid, read_word_ready = 1;
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

    logic [31:0] accepted_addr [0:7];
    integer command_count = 0;
    integer response_count = 0;
    integer output_count = 0;
    integer last_output_cycle = -1;
    integer cycle_count = 0;

    always #5 clk = ~clk;

    function automatic logic [31:0] expected_word(
        input logic [31:0] addr
    );
        expected_word = addr ^ 32'ha5a55a5a;
    endfunction

    memory_io_driver #(
        .READ_QUEUE_DEPTH(8),
        .CACHE_LINES(4)
    ) dut (.*);

    // The memory model exposes the latency-hiding requirement: no response is
    // returned until all eight independent AR-equivalent commands are issued.
    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (!rst_n) begin
            command_count <= 0;
            response_count <= 0;
            mem_read_valid <= 0;
            mem_read_last <= 0;
        end else begin
            if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write) begin
                if (mem_cmd_len != 0)
                    $fatal(1, "strided word used %0d+1 AXI beats",
                           mem_cmd_len);
                accepted_addr[command_count] <= mem_cmd_addr;
                command_count <= command_count + 1;
            end

            if (!mem_read_valid && command_count == 8 &&
                response_count < 8) begin
                mem_read_data <= expected_word(
                    accepted_addr[response_count]);
                mem_read_last <= 1;
                mem_read_valid <= 1;
            end else if (mem_read_valid && mem_read_ready) begin
                response_count <= response_count + 1;
                if (response_count + 1 < 8) begin
                    mem_read_data <= expected_word(
                        accepted_addr[response_count + 1]);
                    mem_read_valid <= 1;
                    mem_read_last <= 1;
                end else begin
                    mem_read_valid <= 0;
                    mem_read_last <= 0;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (read_word_valid && read_word_ready) begin
            if (read_word_data !== expected_word(
                    32'h100 + output_count * 32'h80) ||
                read_word_keep != 4'hf || !read_word_last)
                $fatal(1, "output[%0d]=%h keep=%h last=%0b",
                       output_count, read_word_data,
                       read_word_keep, read_word_last);
            if (last_output_cycle >= 0 &&
                cycle_count != last_output_cycle + 1)
                $fatal(1, "output bubble after word %0d",
                       output_count - 1);
            last_output_cycle <= cycle_count;
            output_count <= output_count + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (int row = 0; row < 8; row++) begin
            @(negedge clk);
            read_req_addr = 32'h100 + row * 32'h80;
            read_req_valid = 1;
            while (!read_req_ready)
                @(negedge clk);
            @(posedge clk);
        end
        @(negedge clk);
        read_req_valid = 0;
        wait (command_count == 8);
        wait (output_count == 8);
        if (error)
            $fatal(1, "memory driver asserted error");
        $display("Memory IO strided-word golden trace passed: eight outstanding single-beat reads, one output/cycle");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout commands=%0d responses=%0d outputs=%0d",
               command_count, response_count, output_count);
    end
endmodule
