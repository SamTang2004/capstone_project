module tb_cnn_tensor_ops_ddr;
    logic clk = 0, rst_n = 0, start;
    logic [1:0] mode;
    logic [31:0] in0, in1, in2, out;
    logic [15:0] n, c0, c1, c2, ih, iw, oh, ow;
    logic [7:0] kh, kw, sh, sw, ph, pw;
    logic busy, done, error, read_req_valid, read_req_ready, read_data_valid, read_data_ready, write_req_valid, write_req_ready;
    logic [31:0] read_req_addr, write_req_addr, write_req_data;
    logic [7:0] read_data;
    logic [3:0] write_req_strb;
    logic [7:0] mem [0:4095];
    logic read_pending;
    integer cycles, start_cycle, first_request_cycle;
    logic [31:0] read_address_q;

    always #5 clk = ~clk;
    assign read_req_ready = !read_pending;
    assign write_req_ready = 1'b1;

    cnn_tensor_ops_ddr dut (.*,
        .input0_base_addr(in0), .input1_base_addr(in1), .input2_base_addr(in2), .output_base_addr(out),
        .channels0(c0), .channels1(c1), .channels2(c2), .input_h(ih), .input_w(iw), .output_h(oh), .output_w(ow),
        .kernel_h(kh), .kernel_w(kw), .stride_h(sh), .stride_w(sw), .pad_h(ph), .pad_w(pw));

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (start) begin start_cycle <= cycles; first_request_cycle <= -1; end
        if (read_req_valid && read_req_ready && first_request_cycle < 0) first_request_cycle <= cycles;
        read_data_valid <= 0;
        if (read_pending) begin
            read_data <= mem[read_address_q];
            read_data_valid <= 1;
            if (read_data_ready) read_pending <= 0;
        end
        if (read_req_valid && read_req_ready) begin
            read_address_q <= read_req_addr;
            read_pending <= 1;
        end
        if (write_req_valid && write_req_ready) begin
            for (int lane = 0; lane < 4; lane++)
                if (write_req_strb[lane]) mem[write_req_addr + lane] <= write_req_data[lane*8 +: 8];
        end
    end

    task automatic clear_memory;
        for (int i = 0; i < 4096; i++) mem[i] = 0;
    endtask

    task automatic launch;
        integer expected_latency;
        begin
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            while (!done) @(posedge clk);
            if (error) $fatal(1, "CNN tensor operator reported error");
            expected_latency = mode == 1 ? 17 : 14;
            if (first_request_cycle - start_cycle < expected_latency)
                $fatal(1, "CNN tensor pipeline latency was %0d cycles, expected at least %0d", first_request_cycle - start_cycle, expected_latency);
            @(posedge clk);
        end
    endtask

    task automatic expect_byte(input int address, input logic signed [7:0] expected);
        if ($signed(mem[address]) !== expected) $fatal(1, "byte mismatch at %0d got %0d expected %0d", address, $signed(mem[address]), expected);
    endtask

    initial begin
        cycles = 0; start_cycle = -1; first_request_cycle = -1;
        start = 0; mode = 0; in0 = 32'h100; in1 = 32'h200; in2 = 32'h300; out = 32'h400;
        n = 1; c0 = 1; c1 = 0; c2 = 0; ih = 0; iw = 0; oh = 0; ow = 0; kh = 0; kw = 0; sh = 0; sw = 0; ph = 0; pw = 0;
        read_pending = 0; read_data_valid = 0; read_data = 0; read_address_q = 0;
        clear_memory();
        repeat (3) @(posedge clk); rst_n = 1;

        // 2x2 max pool over a signed 3x3 NCHW input.
        mem[16'h100] = 1; mem[16'h101] = -2; mem[16'h102] = 3;
        mem[16'h103] = 4; mem[16'h104] = 5; mem[16'h105] = -6;
        mem[16'h106] = -7; mem[16'h107] = 8; mem[16'h108] = 9;
        mode = 1; n = 1; c0 = 1; c1 = 0; c2 = 0; ih = 3; iw = 3; oh = 2; ow = 2; kh = 2; kw = 2; sh = 1; sw = 1; ph = 0; pw = 0;
        launch();
        expect_byte(16'h400, 5); expect_byte(16'h401, 5); expect_byte(16'h402, 8); expect_byte(16'h403, 9);

        // Exact 2x nearest-neighbor upsample, with no runtime division in RTL.
        clear_memory(); mem[16'h100] = 1; mem[16'h101] = 2; mem[16'h102] = 3; mem[16'h103] = 4;
        mode = 2; n = 1; c0 = 1; ih = 2; iw = 2; oh = 4; ow = 4; kh = 0; kw = 0; sh = 2; sw = 2; ph = 0; pw = 0;
        launch();
        expect_byte(16'h400, 1); expect_byte(16'h401, 1); expect_byte(16'h402, 2); expect_byte(16'h403, 2);
        expect_byte(16'h404, 1); expect_byte(16'h405, 1); expect_byte(16'h406, 2); expect_byte(16'h407, 2);
        expect_byte(16'h408, 3); expect_byte(16'h409, 3); expect_byte(16'h40a, 4); expect_byte(16'h40b, 4);
        expect_byte(16'h40c, 3); expect_byte(16'h40d, 3); expect_byte(16'h40e, 4); expect_byte(16'h40f, 4);

        // Three channel-contiguous NCHW inputs concatenate to a single tensor.
        clear_memory(); mem[16'h100] = 1; mem[16'h101] = 2; mem[16'h200] = 3; mem[16'h201] = 4; mem[16'h300] = 5; mem[16'h301] = 6;
        mode = 3; n = 1; c0 = 1; c1 = 1; c2 = 1; ih = 1; iw = 2; oh = 1; ow = 2; kh = 0; kw = 0; sh = 0; sw = 0; ph = 0; pw = 0;
        launch();
        expect_byte(16'h400, 1); expect_byte(16'h401, 2); expect_byte(16'h402, 3); expect_byte(16'h403, 4); expect_byte(16'h404, 5); expect_byte(16'h405, 6);

        // Two batches exercise the pipelined batch-multiply and the second
        // concat source.  The input buffers are independently NCHW packed.
        clear_memory();
        mem[16'h100] = 10; mem[16'h101] = 11; mem[16'h102] = 12; mem[16'h103] = 13;
        mem[16'h200] = 20; mem[16'h201] = 21; mem[16'h202] = 22; mem[16'h203] = 23;
        mem[16'h300] = 30; mem[16'h301] = 31; mem[16'h302] = 32; mem[16'h303] = 33;
        mode = 3; n = 2; c0 = 1; c1 = 1; c2 = 1; ih = 1; iw = 2; oh = 1; ow = 2; kh = 0; kw = 0; sh = 0; sw = 0; ph = 0; pw = 0;
        launch();
        expect_byte(16'h400, 10); expect_byte(16'h401, 11); expect_byte(16'h402, 20); expect_byte(16'h403, 21); expect_byte(16'h404, 30); expect_byte(16'h405, 31);
        expect_byte(16'h406, 12); expect_byte(16'h407, 13); expect_byte(16'h408, 22); expect_byte(16'h409, 23); expect_byte(16'h40a, 32); expect_byte(16'h40b, 33);
        $display("tb_cnn_tensor_ops_ddr golden trace passed");
        $finish;
    end
endmodule
