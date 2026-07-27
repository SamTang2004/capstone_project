`timescale 1ns/1ps
module tb_memory_io_driver_multiread;
    logic clk = 0;
    logic rst_n = 0;
    logic invalidate_cache = 0;
    logic read_req_valid = 0, read_req_ready;
    logic [31:0] read_req_addr;
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
    integer accepted = 0;
    integer received = 0;
    integer read_bursts = 0;
    integer beat = 0;
    integer memory_delay = 0;
    logic burst_pending = 0;
    logic [7:0] data_bias = 0;

    always #5 clk = ~clk;
    function automatic logic [31:0] line_word(input integer index);
        line_word = {8'(index*4+3+data_bias), 8'(index*4+2+data_bias),
                     8'(index*4+1+data_bias), 8'(index*4+data_bias)};
    endfunction

    memory_io_driver dut (.*);

    always_ff @(posedge clk) begin
        if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write) begin
            $display("read burst accepted at %0t", $time);
            if (mem_cmd_addr != 32'h100 || mem_cmd_len != 15)
                $fatal(1, "unexpected line request addr=%h len=%0d",
                       mem_cmd_addr, mem_cmd_len);
            read_bursts <= read_bursts + 1;
            beat <= 0;
            burst_pending <= 1;
            memory_delay <= 30;
        end else if (burst_pending && memory_delay != 0) begin
            memory_delay <= memory_delay - 1;
        end else if (burst_pending && !mem_read_valid) begin
            burst_pending <= 0;
            mem_read_valid <= 1;
            mem_read_data <= line_word(0);
            mem_read_last <= 0;
        end else if (mem_read_valid && mem_read_ready) begin
            if (beat == 15) begin
                mem_read_valid <= 0;
                mem_read_last <= 0;
            end else begin
                beat <= beat + 1;
                mem_read_data <= line_word(beat + 1);
                mem_read_last <= beat == 14;
            end
        end

        if (read_req_valid && read_req_ready)
            accepted <= accepted + 1;
        if (read_data_valid && read_data_ready) begin
            $display("response %0d data=%0d at %0t",
                     received, read_data, $time);
            if (read_data !==
                    (received < 8 ? received[7:0] : data_bias) ||
                !read_data_last)
                $fatal(1, "response[%0d]=%0d last=%0d",
                       received, read_data, read_data_last);
            received <= received + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (int request = 0; request < 8; request++) begin
            @(negedge clk);
            if (!read_req_ready)
                $fatal(1, "read queue backpressured request %0d", request);
            read_req_addr = 32'h100 + request;
            read_req_valid = 1;
            @(posedge clk);
        end
        @(negedge clk);
        read_req_valid = 0;
        $display("queued requests accepted=%0d busy=%0d error=%0d",
                 accepted, busy, error);
        if (accepted != 8)
            $fatal(1, "queue accepted %0d requests", accepted);
        wait (received == 8);
        if (read_bursts != 1)
            $fatal(1, "eight queued same-line reads used %0d bursts",
                   read_bursts);
        wait (!busy);
        @(negedge clk);
        invalidate_cache = 1;
        data_bias = 8'd64;
        @(negedge clk);
        invalidate_cache = 0;
        read_req_addr = 32'h100;
        read_req_valid = 1;
        @(posedge clk);
        @(negedge clk);
        read_req_valid = 0;
        wait (received == 9);
        if (read_bursts != 2)
            $fatal(1, "cache invalidation did not force a second AXI burst");
        if (error)
            $fatal(1, "memory driver asserted error");
        $display("Memory IO eight-read queue/reorder golden trace passed: one shared burst plus one command-boundary refetch");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
