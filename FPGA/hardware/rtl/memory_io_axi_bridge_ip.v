module memory_io_axi_bridge_ip (
    input wire aclk, input wire aresetn,
    input wire cmd_valid, output wire cmd_ready, input wire cmd_write, input wire [31:0] cmd_addr, input wire [3:0] cmd_len,
    input wire [31:0] cmd_wdata, input wire [3:0] cmd_wstrb, input wire write_valid, output wire write_ready,
    input wire [31:0] write_data, input wire [3:0] write_strb, input wire write_last, output wire read_valid, output wire read_last, input wire read_ready,
    output wire [31:0] read_data, output wire error,
    output wire [31:0] m_axi_awaddr, output wire [7:0] m_axi_awlen, output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst, output wire m_axi_awvalid, input wire m_axi_awready,
    output wire [31:0] m_axi_wdata, output wire [3:0] m_axi_wstrb, output wire m_axi_wlast, output wire m_axi_wvalid,
    input wire m_axi_wready, input wire [1:0] m_axi_bresp, input wire m_axi_bvalid, output wire m_axi_bready,
    output wire [31:0] m_axi_araddr, output wire [7:0] m_axi_arlen, output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst, output wire m_axi_arvalid, input wire m_axi_arready,
    input wire [31:0] m_axi_rdata, input wire [1:0] m_axi_rresp, input wire m_axi_rvalid, input wire m_axi_rlast, output wire m_axi_rready
);
    memory_io_axi_bridge bridge (
        .aclk(aclk), .aresetn(aresetn), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_write(cmd_write),
        .cmd_addr(cmd_addr), .cmd_len(cmd_len), .cmd_wdata(cmd_wdata), .cmd_wstrb(cmd_wstrb),
        .write_valid(write_valid), .write_ready(write_ready), .write_data(write_data), .write_strb(write_strb), .write_last(write_last), .read_valid(read_valid), .read_last(read_last), .read_ready(read_ready),
        .read_data(read_data), .error(error),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rvalid(m_axi_rvalid), .m_axi_rlast(m_axi_rlast), .m_axi_rready(m_axi_rready)
    );
endmodule
