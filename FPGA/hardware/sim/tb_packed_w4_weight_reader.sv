module tb_packed_w4_weight_reader;
    logic clk = 0;
    logic rst_n = 0;
    logic start;
    logic [31:0] base_addr, weight_count;
    logic busy, done, error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [6:0] read_req_len;
    logic read_req_wide, read_req_qword, read_req_allow_ahead;
    logic read_qword_valid, read_qword_ready, read_qword_last;
    logic [63:0] read_qword_data;
    logic [7:0] read_qword_keep;
    logic weight_valid, weight_ready, weight_last;
    logic [31:0] packed_weights;
    logic [7:0] weight_lane_valid;
    logic [7:0] memory [0:255];
    integer errors = 0;
    integer request_count = 0;
    integer request_addr_log [0:15];
    integer request_len_log [0:15];

    always #5 clk = ~clk;

    packed_w4_weight_reader dut (.*);

    initial begin
        #100000;
        $fatal(1, "packed W4 reader timeout");
    end

    // A deliberately minimal cache-bus model: one request is accepted, then
    // its compact 64-bit response words are returned under backpressure.
    initial begin : cache_model
        integer addr;
        integer remaining;
        integer beat_bytes;
        read_req_ready = 0;
        read_qword_valid = 0;
        read_qword_data = 0;
        read_qword_keep = 0;
        read_qword_last = 0;
        wait (rst_n);
        forever begin
            wait (read_req_valid);
            @(negedge clk);
            read_req_ready = 1;
            @(posedge clk);
            addr = read_req_addr;
            remaining = read_req_len;
            request_addr_log[request_count] = read_req_addr;
            request_len_log[request_count] = read_req_len;
            request_count++;
            @(negedge clk);
            read_req_ready = 0;
            while (remaining > 0) begin
                beat_bytes = remaining > 8 ? 8 : remaining;
                read_qword_data = 0;
                read_qword_keep = 0;
                for (int lane = 0; lane < beat_bytes; lane++) begin
                    read_qword_data[lane*8 +: 8] = memory[addr + lane];
                    read_qword_keep[lane] = 1'b1;
                end
                read_qword_last = remaining <= 8;
                read_qword_valid = 1;
                do @(posedge clk); while (!read_qword_ready);
                addr += beat_bytes;
                remaining -= beat_bytes;
                @(negedge clk);
                read_qword_valid = 0;
                read_qword_last = 0;
            end
        end
    end

    task automatic run_case(input integer base, input integer count);
        integer vectors;
        logic [31:0] expected_data;
        logic [7:0] expected_valid;
        begin
            for (int byte_index = 0; byte_index < (count + 1) / 2; byte_index++)
                memory[base + byte_index] = (((byte_index*2 + 1) & 15) << 4) |
                                             ((byte_index*2) & 15);

            @(negedge clk);
            base_addr = base;
            weight_count = count;
            start = 1;
            @(negedge clk);
            start = 0;
            vectors = (count + 7) / 8;

            for (int vector_index = 0; vector_index < vectors; vector_index++) begin
                expected_data = 0;
                expected_valid = 0;
                for (int lane = 0; lane < 8; lane++) begin
                    if (vector_index*8 + lane < count) begin
                        expected_data[lane*4 +: 4] = (vector_index*8 + lane) & 15;
                        expected_valid[lane] = 1'b1;
                    end
                end
                weight_ready = vector_index[0] ? 1'b0 : 1'b1;
                repeat (vector_index[0] ? 2 : 0) @(posedge clk);
                weight_ready = 1'b1;
                wait (weight_valid);
                #1;
                if (packed_weights !== expected_data || weight_lane_valid !== expected_valid ||
                    weight_last !== (vector_index == vectors-1)) begin
                    $error("vector=%0d data=%h/%h lanes=%h/%h last=%0b", vector_index,
                           packed_weights, expected_data, weight_lane_valid, expected_valid, weight_last);
                    errors++;
                end
                @(posedge clk);
                #1;
                weight_ready = 0;
            end
            wait (done);
            @(posedge clk);
        end
    endtask

    initial begin
        start = 0;
        base_addr = 0;
        weight_count = 0;
        weight_ready = 0;
        for (int i = 0; i < 256; i++) memory[i] = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        // 31 weights exercise a partially valid final nibble vector.
        run_case(32, 31);
        // Starting at byte 60 forces legal 4-byte and 16-byte cache-line reads.
        request_count = 0;
        run_case(60, 40);
        if (request_count != 2 || request_addr_log[0] != 60 || request_len_log[0] != 4 ||
            request_addr_log[1] != 64 || request_len_log[1] != 16) begin
            $error("cache-line split mismatch count=%0d req0=%0d/%0d req1=%0d/%0d",
                   request_count, request_addr_log[0], request_len_log[0],
                   request_addr_log[1], request_len_log[1]);
            errors++;
        end

        @(negedge clk);
        base_addr = 2;
        weight_count = 8;
        start = 1;
        @(negedge clk);
        start = 0;
        wait (done);
        if (!error) begin
            $error("unaligned base address was not rejected");
            errors++;
        end

        if (errors) $fatal(1, "packed W4 reader failed with %0d errors", errors);
        $display("Packed W4 wide-reader golden trace passed");
        $finish;
    end
endmodule
