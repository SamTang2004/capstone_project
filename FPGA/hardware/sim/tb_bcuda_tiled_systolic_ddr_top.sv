module tb_bcuda_tiled_systolic_ddr_top;
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_A.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_B.hex";
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_C.hex";
    localparam int MEM_BYTES = 4096;
    localparam int A_BASE = 32;
    localparam int B_BASE = 1024;
    localparam int C_BASE = 2048;
    localparam int A_STRIDE = 16;
    localparam int B_STRIDE = 16;
    localparam int C_STRIDE = 64;
    localparam int RQ_C_BASE = 3072;
    localparam int CONV_A_BASE = 64;
    localparam int CONV_B_BASE = 512;
    localparam int CONV_C_BASE = 1536;
    localparam int CONV_BIAS_BASE = 2560;
    localparam int CONV_RQ_BASE = 3104;
    localparam int CONV_RELU_BASE = 3136;
    localparam int TENSOR0_BASE = 3200;
    localparam int TENSOR1_BASE = 3264;
    localparam int TENSOR2_BASE = 3328;
    localparam int TENSOR_OUT_BASE = 3456;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [5:0] awaddr;
    logic awvalid;
    logic awready;
    logic [31:0] wdata;
    logic [3:0] wstrb;
    logic wvalid;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready;
    logic [5:0] araddr;
    logic arvalid;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready;
    logic mem_cmd_valid;
    logic mem_cmd_ready;
    logic mem_cmd_write;
    logic [31:0] mem_cmd_addr;
    logic [3:0] mem_cmd_len;
    logic [31:0] mem_cmd_wdata;
    logic [3:0] mem_cmd_wstrb;
    logic mem_write_valid;
    logic mem_write_ready;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_write_last;
    logic mem_read_valid;
    logic mem_read_ready;
    logic [31:0] mem_read_data;
    logic mem_read_last;

    logic [7:0] ddr [0:MEM_BYTES-1];
    logic [7:0] a_mem [0:255];
    logic [7:0] b_mem [0:255];
    logic [31:0] c_mem [0:255];
    logic pending_read;
    logic [31:0] pending_read_addr;
    logic [4:0] pending_read_remaining;
    logic [31:0] pending_write_addr;
    logic [1:0] ddr_phase;
    int errors;
    integer read_cmd_count;
    integer write_cmd_count;
    integer nonzero_write_count;

    always #5 clk = ~clk;

    initial begin
        repeat (40000) @(posedge clk);
        $fatal(1, "Tiled DDR top timeout state=%0d rq_busy=%0b read_pending=%0b count=%0d flush=%0b send=%0b inflight=%0b", dut.requantizer.state, dut.rq_busy, dut.memio.read_pending, dut.memio.write_count, dut.memio.write_flush_pending, dut.memio.write_send_active, dut.memio.write_inflight);
    end

    bcuda_tiled_systolic_ddr_top dut (
        .aclk(clk),
        .aresetn(rst_n),
        .s_axi_awaddr(awaddr),
        .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),
        .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),
        .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid),
        .s_axi_bready(bready),
        .s_axi_araddr(araddr),
        .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),
        .s_axi_rdata(rdata),
        .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid),
        .s_axi_rready(rready),
        .mem_cmd_valid(mem_cmd_valid),
        .mem_cmd_ready(mem_cmd_ready),
        .mem_cmd_write(mem_cmd_write),
        .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_len(mem_cmd_len),
        .mem_cmd_wdata(mem_cmd_wdata),
        .mem_cmd_wstrb(mem_cmd_wstrb),
        .mem_write_valid(mem_write_valid),
        .mem_write_ready(mem_write_ready),
        .mem_write_data(mem_write_data),
        .mem_write_strb(mem_write_strb),
        .mem_write_last(mem_write_last),
        .mem_read_valid(mem_read_valid),
        .mem_read_ready(mem_read_ready),
        .mem_read_data(mem_read_data),
        .mem_read_last(mem_read_last)
    );

    function automatic logic [31:0] load_word_le(input logic [31:0] addr);
        begin
            load_word_le = {ddr[addr + 3], ddr[addr + 2], ddr[addr + 1], ddr[addr]};
        end
    endfunction

    task automatic store_word_le(input logic [31:0] addr, input logic [31:0] data, input logic [3:0] strobe);
        begin
            for (int lane = 0; lane < 4; lane++) begin
                if (strobe[lane]) begin
                    ddr[addr + lane] = data[8*lane +: 8];
                end
            end
        end
    endtask

    task automatic axi_write(input logic [5:0] addr, input logic [31:0] data);
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            wstrb = 4'hF;
            awvalid = 1'b1;
            wvalid = 1'b1;
            bready = 1'b1;
            @(posedge clk);
            while (!bvalid) @(posedge clk);
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [5:0] addr, output logic [31:0] data);
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1'b1;
            rready = 1'b1;
            @(posedge clk);
            while (!rvalid) @(posedge clk);
            data = rdata;
            @(negedge clk);
            arvalid = 1'b0;
            rready = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            mem_cmd_ready <= 1'b0;
            mem_write_ready <= 1'b0;
            mem_read_valid <= 1'b0;
            mem_read_data <= '0;
            pending_read <= 1'b0;
            pending_read_addr <= '0;
            pending_read_remaining <= '0;
            pending_write_addr <= '0;
            mem_read_last <= 1'b0;
            ddr_phase <= '0;
        end else begin
            // Model ordinary HP0/interconnect stalls.  The command channel
            // pauses one in four cycles and write data every other cycle.
            ddr_phase <= ddr_phase + 1'b1;
            mem_cmd_ready <= ddr_phase != 2'd0;
            mem_write_ready <= ddr_phase[0];
            mem_read_valid <= pending_read;
            mem_read_data <= load_word_le(pending_read_addr);
            mem_read_last <= pending_read && pending_read_remaining == 1;
            if (pending_read && mem_read_ready) begin
                if (pending_read_remaining == 1) pending_read <= 1'b0;
                else begin pending_read_addr <= pending_read_addr + 4; pending_read_remaining <= pending_read_remaining - 1'b1; end
            end

            if (mem_cmd_valid && mem_cmd_ready) begin
                if (mem_cmd_write) begin
                    write_cmd_count = write_cmd_count + 1;
                    if (mem_cmd_addr[1:0] != 2'b00) begin
                        $error("Tiled DDR GEMM top emitted unaligned write command addr=0x%08h", mem_cmd_addr);
                        errors++;
                    end
                    pending_write_addr <= mem_cmd_addr;
                end else begin
                    if (mem_cmd_addr[1:0] != 2'b00) begin
                        $error("Tiled DDR GEMM top emitted unaligned read command addr=0x%08h", mem_cmd_addr);
                        errors++;
                    end
                    pending_read <= 1'b1;
                    pending_read_addr <= mem_cmd_addr;
                    pending_read_remaining <= mem_cmd_len + 1'b1;
                    read_cmd_count = read_cmd_count + 1;
                end
            end
            if (mem_write_valid && mem_write_ready) begin
                if (mem_write_data != 32'h00000000) nonzero_write_count = nonzero_write_count + 1;
                store_word_le(pending_write_addr, mem_write_data, mem_write_strb);
                pending_write_addr <= pending_write_addr + 4;
            end
        end
    end

    initial begin
        logic [31:0] status;
        errors = 0;
        read_cmd_count = 0;
        write_cmd_count = 0;
        nonzero_write_count = 0;
        $readmemh(A_PATH, a_mem);
        $readmemh(B_PATH, b_mem);
        $readmemh(C_PATH, c_mem);

        for (int i = 0; i < MEM_BYTES; i++) begin
            ddr[i] = 8'hA5;
        end
        for (int i = 0; i < 256; i++) begin
            ddr[A_BASE + i] = a_mem[i];
            ddr[B_BASE + i] = b_mem[i];
        end

        awaddr = '0;
        awvalid = 1'b0;
        wdata = '0;
        wstrb = 4'h0;
        wvalid = 1'b0;
        bready = 1'b0;
        araddr = '0;
        arvalid = 1'b0;
        rready = 1'b0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        axi_write(6'h08, A_BASE);
        axi_write(6'h0C, B_BASE);
        axi_write(6'h10, C_BASE);
        axi_write(6'h14, A_STRIDE);
        axi_write(6'h18, B_STRIDE);
        axi_write(6'h1C, C_STRIDE);
        axi_write(6'h20, 32'd16);
        axi_write(6'h24, 32'd16);
        axi_write(6'h28, 32'd16);
        axi_write(6'h2C, 32'd16);
        axi_write(6'h30, 32'd1);
        axi_write(6'h00, 32'h1);

        repeat (5000) begin
            axi_read(6'h04, status);
            if (status[1]) begin
                break;
            end
        end

        if (!status[1] || status[2]) begin
            $fatal(1, "Tiled DDR GEMM top did not complete cleanly, status=0x%08h", status);
        end

        if (read_cmd_count != 9) begin
            $error("Tiled DDR GEMM top expected nine 16-word A/B line fills, saw %0d", read_cmd_count);
            errors++;
        end
        if (write_cmd_count != 16) begin
            $error("Tiled DDR GEMM top expected sixteen 16-word C write bursts, saw %0d", write_cmd_count);
            errors++;
        end
        if (nonzero_write_count == 0) begin
            $error("Tiled DDR GEMM top wrote only zero C values; check tile load, core start, or stale/partial compile set");
            errors++;
        end

        for (int i = 0; i < 256; i++) begin
            logic [31:0] got;
            got = load_word_le(C_BASE + i * 4);
            if (got !== c_mem[i]) begin
                $error("Tiled DDR GEMM C mismatch index %0d expected=0x%08h got=0x%08h", i, c_mem[i], got);
                errors++;
            end
        end

        // Regression for the board failure where a new GEMM began with two
        // words retained from the preceding result tile.  Dirty C first, then
        // issue the same GEMM again without resetting the PL datapath.
        for (int i = 0; i < 16 * 16 * 4; i++) begin
            ddr[C_BASE + i] = 8'hA5;
        end
        axi_write(6'h00, 32'h0);
        axi_write(6'h00, 32'h1);
        repeat (5000) begin
            axi_read(6'h04, status);
            if (status[1]) begin
                break;
            end
        end
        if (!status[1] || status[2]) begin
            $fatal(1, "Back-to-back tiled DDR GEMM did not complete cleanly, status=0x%08h", status);
        end
        for (int i = 0; i < 256; i++) begin
            logic [31:0] got;
            got = load_word_le(C_BASE + i * 4);
            if (got !== c_mem[i]) begin
                $error("Back-to-back tiled DDR GEMM C mismatch index %0d expected=0x%08h got=0x%08h", i, c_mem[i], got);
                errors++;
            end
        end

        /* Verify the PL requantize command shares the same DDR path. */
        ddr[A_BASE] = 8'hd4; ddr[A_BASE + 1] = 8'hfe; ddr[A_BASE + 2] = 8'hff; ddr[A_BASE + 3] = 8'hff;
        ddr[A_BASE + 4] = 8'hfe; ddr[A_BASE + 5] = 8'hff; ddr[A_BASE + 6] = 8'hff; ddr[A_BASE + 7] = 8'hff;
        ddr[A_BASE + 8] = 8'h00; ddr[A_BASE + 9] = 8'h00; ddr[A_BASE + 10] = 8'h00; ddr[A_BASE + 11] = 8'h00;
        ddr[A_BASE + 12] = 8'h08; ddr[A_BASE + 13] = 8'h00; ddr[A_BASE + 14] = 8'h00; ddr[A_BASE + 15] = 8'h00;
        ddr[A_BASE + 16] = 8'hf4; ddr[A_BASE + 17] = 8'h01; ddr[A_BASE + 18] = 8'h00; ddr[A_BASE + 19] = 8'h00;
        ddr[A_BASE + 20] = 8'he8; ddr[A_BASE + 21] = 8'h03; ddr[A_BASE + 22] = 8'h00; ddr[A_BASE + 23] = 8'h00;
        ddr[A_BASE + 24] = 8'h18; ddr[A_BASE + 25] = 8'hfc; ddr[A_BASE + 26] = 8'hff; ddr[A_BASE + 27] = 8'hff;
        axi_write(6'h00, 32'h0);
        axi_write(6'h08, A_BASE);
        axi_write(6'h10, RQ_C_BASE);
        axi_write(6'h20, 32'd7);
        axi_write(6'h34, 32'd1);
        axi_write(6'h38, 32'd1);
        axi_write(6'h3C, 32'd3);
        axi_write(6'h00, 32'h3);
        repeat (300) begin
            axi_read(6'h04, status);
            if (status[1]) break;
        end
        if (!status[1] || status[2]) begin
            $error("Tiled DDR requantize command did not complete cleanly, status=0x%08h", status);
            errors++;
        end
        if ($signed(ddr[RQ_C_BASE]) != -128 || $signed(ddr[RQ_C_BASE + 1]) != 2 || $signed(ddr[RQ_C_BASE + 2]) != 3 ||
            $signed(ddr[RQ_C_BASE + 3]) != 7 || $signed(ddr[RQ_C_BASE + 4]) != 127 || $signed(ddr[RQ_C_BASE + 5]) != 127 ||
            $signed(ddr[RQ_C_BASE + 6]) != -128 || ddr[RQ_C_BASE + 7] != 8'hA5) begin
            $error("Tiled DDR requantize results or final byte strobe mismatch");
            errors++;
        end

        /* Full PL path: NCHW/OIHW Conv2D with INT32 bias -> requantize -> ReLU. */
        for (int i = 0; i < 9; i++) ddr[CONV_A_BASE + i] = i + 1;
        ddr[CONV_B_BASE] = 8'sd1; ddr[CONV_B_BASE + 1] = 8'sd0;
        ddr[CONV_B_BASE + 2] = 8'sd0; ddr[CONV_B_BASE + 3] = 8'sd1;
        ddr[CONV_BIAS_BASE] = 8'hf6; ddr[CONV_BIAS_BASE + 1] = 8'hff;
        ddr[CONV_BIAS_BASE + 2] = 8'hff; ddr[CONV_BIAS_BASE + 3] = 8'hff;
        axi_write(6'h00, 32'h0);
        axi_write(6'h08, CONV_A_BASE);
        axi_write(6'h0C, CONV_B_BASE);
        axi_write(6'h10, CONV_C_BASE);
        axi_write(6'h14, 32'd1);       // input C
        axi_write(6'h18, 32'd3);       // input H
        axi_write(6'h1C, 32'd3);       // input W
        axi_write(6'h20, 32'd1);       // output C
        axi_write(6'h24, 32'd2);       // output H
        axi_write(6'h28, 32'd2);       // output W
        axi_write(6'h2C, 32'd2);       // kernel H
        axi_write(6'h30, 32'd2);       // kernel W
        axi_write(6'h34, 32'h00000101); // stride H/W
        axi_write(6'h38, 32'd0);       // padding H/W
        axi_write(6'h3C, CONV_BIAS_BASE);
        axi_write(6'h00, 32'h5);       // START | CONV2D
        repeat (5000) begin
            axi_read(6'h04, status);
            if (status[1]) break;
        end
        if (!status[1] || status[2] || $signed(load_word_le(CONV_C_BASE)) != -4 ||
            $signed(load_word_le(CONV_C_BASE + 4)) != -2 || $signed(load_word_le(CONV_C_BASE + 8)) != 2 ||
            $signed(load_word_le(CONV_C_BASE + 12)) != 4) begin
            $error("Biased PL Conv2D result mismatch status=0x%08h got=[%0d,%0d,%0d,%0d]", status,
                   $signed(load_word_le(CONV_C_BASE)), $signed(load_word_le(CONV_C_BASE + 4)),
                   $signed(load_word_le(CONV_C_BASE + 8)), $signed(load_word_le(CONV_C_BASE + 12)));
            $display("Accumulator entries after Conv: %0d %0d %0d %0d", $signed(dut.accumulator_banks[0].accumulator.mem[0]),
                     $signed(dut.accumulator_banks[0].accumulator.mem[16]), $signed(dut.accumulator_banks[0].accumulator.mem[32]),
                     $signed(dut.accumulator_banks[0].accumulator.mem[48]));
            $display("Bias preload geometry rows=%0d cols=%0d", dut.bias_preloader.rows_r, dut.bias_preloader.cols_r);
            errors++;
        end

        axi_write(6'h00, 32'h0);
        axi_write(6'h08, CONV_C_BASE);
        axi_write(6'h10, CONV_RQ_BASE);
        axi_write(6'h20, 32'd4);
        axi_write(6'h34, 32'd1);
        axi_write(6'h38, 32'd0);
        axi_write(6'h3C, 32'd0);
        axi_write(6'h00, 32'h3);       // START | REQUANTIZE
        repeat (300) begin axi_read(6'h04, status); if (status[1]) break; end
        axi_write(6'h00, 32'h0);
        axi_write(6'h08, CONV_RQ_BASE);
        axi_write(6'h10, CONV_RELU_BASE);
        axi_write(6'h20, 32'd4);
        axi_write(6'h38, 32'd0);
        axi_write(6'h00, 32'h9);       // START | ACTIVATION (ReLU)
        repeat (300) begin axi_read(6'h04, status); if (status[1]) break; end
        if (!status[1] || status[2] || $signed(ddr[CONV_RELU_BASE]) != 0 || $signed(ddr[CONV_RELU_BASE + 1]) != 0 ||
            $signed(ddr[CONV_RELU_BASE + 2]) != 2 || $signed(ddr[CONV_RELU_BASE + 3]) != 4) begin
            $error("PL Conv2D requantize/ReLU chain mismatch status=0x%08h", status);
            errors++;
        end

        /* MaxPool, integer nearest upsample, and concat use the same HP0 path. */
        ddr[TENSOR0_BASE] = 1; ddr[TENSOR0_BASE + 1] = -2; ddr[TENSOR0_BASE + 2] = 3;
        ddr[TENSOR0_BASE + 3] = 4; ddr[TENSOR0_BASE + 4] = 5; ddr[TENSOR0_BASE + 5] = -6;
        ddr[TENSOR0_BASE + 6] = -7; ddr[TENSOR0_BASE + 7] = 8; ddr[TENSOR0_BASE + 8] = 9;
        axi_write(6'h00, 0); axi_write(6'h08, TENSOR0_BASE); axi_write(6'h10, TENSOR_OUT_BASE);
        axi_write(6'h14, 1); axi_write(6'h18, 1); axi_write(6'h1C, 3); axi_write(6'h20, 3); axi_write(6'h24, 2); axi_write(6'h28, 2);
        axi_write(6'h2C, 2); axi_write(6'h30, 2); axi_write(6'h34, 32'h00000101); axi_write(6'h38, 0); axi_write(6'h00, 32'h21);
        repeat (5000) begin axi_read(6'h04, status); if (status[1]) break; end
        if (!status[1] || status[2] || $signed(ddr[TENSOR_OUT_BASE]) != 5 || $signed(ddr[TENSOR_OUT_BASE + 1]) != 5 ||
            $signed(ddr[TENSOR_OUT_BASE + 2]) != 8 || $signed(ddr[TENSOR_OUT_BASE + 3]) != 9) begin
            $error("PL MaxPool mismatch status=0x%08h", status); errors++;
        end

        ddr[TENSOR0_BASE] = 1; ddr[TENSOR0_BASE + 1] = 2; ddr[TENSOR0_BASE + 2] = 3; ddr[TENSOR0_BASE + 3] = 4;
        axi_write(6'h00, 0); axi_write(6'h08, TENSOR0_BASE); axi_write(6'h10, TENSOR_OUT_BASE);
        axi_write(6'h14, 1); axi_write(6'h18, 1); axi_write(6'h1C, 2); axi_write(6'h20, 2); axi_write(6'h24, 4); axi_write(6'h28, 4);
        axi_write(6'h34, 32'h00000202); axi_write(6'h00, 32'h41);
        repeat (5000) begin axi_read(6'h04, status); if (status[1]) break; end
        if (!status[1] || status[2] || $signed(ddr[TENSOR_OUT_BASE]) != 1 || $signed(ddr[TENSOR_OUT_BASE + 2]) != 2 ||
            $signed(ddr[TENSOR_OUT_BASE + 8]) != 3 || $signed(ddr[TENSOR_OUT_BASE + 10]) != 4) begin
            $error("PL upsample mismatch status=0x%08h", status); errors++;
        end

        ddr[TENSOR0_BASE] = 1; ddr[TENSOR0_BASE + 1] = 2; ddr[TENSOR1_BASE] = 3; ddr[TENSOR1_BASE + 1] = 4; ddr[TENSOR2_BASE] = 5; ddr[TENSOR2_BASE + 1] = 6;
        axi_write(6'h00, 0); axi_write(6'h08, TENSOR0_BASE); axi_write(6'h0C, TENSOR1_BASE); axi_write(6'h10, TENSOR_OUT_BASE); axi_write(6'h3C, TENSOR2_BASE);
        axi_write(6'h14, 1); axi_write(6'h18, 1); axi_write(6'h1C, 1); axi_write(6'h20, 2); axi_write(6'h24, 1); axi_write(6'h28, 2);
        axi_write(6'h34, 1); axi_write(6'h38, 1); axi_write(6'h00, 32'h81);
        repeat (5000) begin axi_read(6'h04, status); if (status[1]) break; end
        if (!status[1] || status[2] || $signed(ddr[TENSOR_OUT_BASE]) != 1 || $signed(ddr[TENSOR_OUT_BASE + 1]) != 2 ||
            $signed(ddr[TENSOR_OUT_BASE + 2]) != 3 || $signed(ddr[TENSOR_OUT_BASE + 3]) != 4 || $signed(ddr[TENSOR_OUT_BASE + 4]) != 5 || $signed(ddr[TENSOR_OUT_BASE + 5]) != 6) begin
            $error("PL concat mismatch status=0x%08h", status); errors++;
        end

        if (errors != 0) begin
            $fatal(1, "Tiled DDR GEMM top golden trace failed with %0d errors", errors);
        end

        $display("Full tiled DDR GEMM, Conv2D, requantize, activation, pool, upsample, and concat golden trace passed");
        $finish;
    end

endmodule
