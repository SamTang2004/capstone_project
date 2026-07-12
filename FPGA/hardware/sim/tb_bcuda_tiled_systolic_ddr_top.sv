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
    int errors;
    integer read_cmd_count;
    integer write_cmd_count;
    integer nonzero_write_count;

    always #5 clk = ~clk;

    initial begin
        repeat (20000) @(posedge clk);
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
        end else begin
            mem_cmd_ready <= 1'b1;
            mem_write_ready <= 1'b1;
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

        repeat (2000) begin
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

        if (errors != 0) begin
            $fatal(1, "Tiled DDR GEMM top golden trace failed with %0d errors", errors);
        end

        $display("Tiled DDR GEMM top golden trace passed: 256 outputs plus requantize");
        $finish;
    end

endmodule
