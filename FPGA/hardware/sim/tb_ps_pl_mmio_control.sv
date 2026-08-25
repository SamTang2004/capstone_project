module tb_ps_pl_mmio_control;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [7:0] awaddr;
    logic awvalid;
    logic awready;
    logic [31:0] wdata;
    logic [3:0] wstrb;
    logic wvalid;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready;
    logic [7:0] araddr;
    logic arvalid;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready;
    logic [31:0] status_word;
    logic [31:0] ctrl_word;
    logic [31:0] debug_ctrl_word;
    logic [32*18-1:0] debug_regs_flat;
    logic [32*6-1:0] extended_perf_regs_flat;
    int errors;

    always #5 clk = ~clk;

    axi_lite_control dut (
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
        .status_word(status_word),
        .perf_regs_flat('0),
        .debug_regs_flat(debug_regs_flat),
        .extended_perf_regs_flat(extended_perf_regs_flat),
        .ctrl_word(ctrl_word),
        .debug_ctrl_word(debug_ctrl_word)
    );

    task automatic axi_write(input logic [7:0] addr, input logic [31:0] data, input logic [3:0] strobe);
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            wstrb = strobe;
            awvalid = 1'b1;
            wvalid = 1'b1;
            bready = 1'b1;
            @(posedge clk);
            while (!bvalid) @(posedge clk);
            if (bresp != 2'b00) begin
                $error("AXI-lite write response error at addr=0x%02h", addr);
                errors++;
            end
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [7:0] addr, output logic [31:0] data);
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1'b1;
            rready = 1'b1;
            while (!rvalid) @(negedge clk);
            data = rdata;
            if (rresp != 2'b00) begin
                $error("AXI-lite read response error at addr=0x%02h", addr);
                errors++;
            end
            @(posedge clk);
            @(negedge clk);
            arvalid = 1'b0;
            rready = 1'b0;
        end
    endtask

    initial begin
        logic [31:0] got;
        errors = 0;
        awaddr = '0;
        awvalid = 1'b0;
        wdata = '0;
        wstrb = 4'h0;
        wvalid = 1'b0;
        bready = 1'b0;
        araddr = '0;
        arvalid = 1'b0;
        rready = 1'b0;
        status_word = 32'h0000_0000;
        debug_regs_flat = '0;
        debug_regs_flat[0 +: 32] = 32'h1234_5678;
        extended_perf_regs_flat = '0;
        extended_perf_regs_flat[5*32 +: 32] = 32'h89ab_cdef;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        axi_write(6'h00, 32'h0000_0001, 4'hF);
        if (ctrl_word !== 32'h0000_0001) begin
            $error("Control word did not latch start bit");
            errors++;
        end

        axi_write(6'h00, 32'h0000_0100, 4'h2);
        if (ctrl_word !== 32'h0000_0101) begin
            $error("Byte strobe update failed: ctrl=0x%08h", ctrl_word);
            errors++;
        end

        status_word = 32'h8000_0005;
        repeat (2) @(posedge clk);
        axi_read(6'h04, got);
        if (got !== 32'h8000_0005) begin
            $error("Status read mismatch expected=0x80000005 got=0x%08h", got);
            errors++;
        end

        axi_write(8'h80, 32'h0000_000d, 4'hF);
        if (debug_ctrl_word !== 32'h0000_000d) begin
            $error("Debug control write mismatch got=0x%08h", debug_ctrl_word);
            errors++;
        end
        axi_read(8'h84, got);
        if (got !== 32'h1234_5678) begin
            $error("Debug counter read mismatch got=0x%08h", got);
            errors++;
        end
        axi_read(8'he0, got);
        if (got !== 32'h89ab_cdef) begin
            $error("Extended performance counter read mismatch got=0x%08h", got);
            errors++;
        end

        if (errors != 0) begin
            $fatal(1, "PS-PL MMIO control test failed with %0d errors", errors);
        end

        $display("PS-PL MMIO control golden trace passed");
        $finish;
    end

endmodule
