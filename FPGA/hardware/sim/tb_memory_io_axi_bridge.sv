module tb_memory_io_axi_bridge;
    logic clk = 0, rst_n = 0, cmd_valid, cmd_ready, cmd_write, read_valid, read_ready, error;
    logic [31:0] cmd_addr, cmd_wdata, read_data;
    logic [3:0] cmd_wstrb;
    logic [31:0] awaddr, wdata, araddr, rdata;
    logic [7:0] awlen, arlen;
    logic [2:0] awsize, arsize;
    logic [1:0] awburst, arburst, bresp, rresp;
    logic awvalid, awready, wlast, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
    logic [31:0] mem [0:15];
    integer errors;
    always #5 clk = ~clk;
    initial begin
        repeat (1000) @(posedge clk);
        $fatal(1, "AXI DDR bridge timeout state=%0d cmd_ready=%0b read_valid=%0b", dut.state, cmd_ready, read_valid);
    end
    assign awready = 1; assign wready = 1; assign arready = 1; assign bresp = 0; assign rresp = 0;
    always_ff @(posedge clk) begin
        if (!rst_n) begin bvalid <= 0; rvalid <= 0; rdata <= 0; end
        else begin
            if (wvalid && wready) mem[awaddr[5:2]] <= wdata;
            bvalid <= wvalid && wready;
            if (bvalid && bready) bvalid <= 0;
            if (arvalid && arready) begin rdata <= mem[araddr[5:2]]; rvalid <= 1; end
            if (rvalid && rready) rvalid <= 0;
        end
    end
    memory_io_axi_bridge dut (
        .aclk(clk), .aresetn(rst_n), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_write(cmd_write), .cmd_addr(cmd_addr),
        .cmd_wdata(cmd_wdata), .cmd_wstrb(cmd_wstrb), .read_valid(read_valid), .read_ready(read_ready), .read_data(read_data), .error(error),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst), .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(), .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready), .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize), .m_axi_arburst(arburst), .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rvalid(rvalid), .m_axi_rready(rready)
    );
    initial begin
        errors=0; cmd_valid=0; cmd_write=0; cmd_addr=0; cmd_wdata=0; cmd_wstrb=0; read_ready=0;
        repeat(2) @(posedge clk); rst_n=1;
        @(negedge clk); cmd_write=1; cmd_addr=32'h10; cmd_wdata=32'h12345678; cmd_wstrb=4'hf; cmd_valid=1;
        @(negedge clk); cmd_valid=0;
        wait(dut.state != 0); wait(dut.state == 0);
        @(negedge clk); cmd_write=0; cmd_addr=32'h10; cmd_valid=1;
        @(negedge clk); cmd_valid=0;
        wait(dut.state != 0); wait(dut.state == 5); wait(rvalid); #1;
        if (read_data !== 32'h12345678 || error || awlen != 0 || awsize != 2 || arlen != 0 || arsize != 2) errors=errors+1;
        read_ready=1;
        if (errors != 0) $fatal(1, "AXI DDR bridge test failed");
        $display("AXI DDR bridge golden trace passed"); $finish;
    end
endmodule
