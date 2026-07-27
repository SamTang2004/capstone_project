module bcuda_tiled_systolic_ddr_ip (
    input wire aclk, input wire aresetn,
    input wire [6:0] s_axi_awaddr, input wire s_axi_awvalid, output wire s_axi_awready,
    input wire [31:0] s_axi_wdata, input wire [3:0] s_axi_wstrb, input wire s_axi_wvalid, output wire s_axi_wready,
    output wire [1:0] s_axi_bresp, output wire s_axi_bvalid, input wire s_axi_bready,
    input wire [6:0] s_axi_araddr, input wire s_axi_arvalid, output wire s_axi_arready,
    output wire [31:0] s_axi_rdata, output wire [1:0] s_axi_rresp, output wire s_axi_rvalid, input wire s_axi_rready,
    output wire mem_cmd_valid, input wire mem_cmd_ready, output wire mem_cmd_write, output wire [31:0] mem_cmd_addr, output wire [3:0] mem_cmd_len,
    output wire [31:0] mem_cmd_wdata, output wire [3:0] mem_cmd_wstrb, output wire mem_write_valid, input wire mem_write_ready,
    output wire [31:0] mem_write_data, output wire [3:0] mem_write_strb, output wire mem_write_last,
    input wire mem_read_valid, output wire mem_read_ready, input wire [31:0] mem_read_data, input wire mem_read_last
);
    bcuda_dual_int4_ddr_top accelerator (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .mem_cmd_valid(mem_cmd_valid), .mem_cmd_ready(mem_cmd_ready), .mem_cmd_write(mem_cmd_write), .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_len(mem_cmd_len), .mem_cmd_wdata(mem_cmd_wdata), .mem_cmd_wstrb(mem_cmd_wstrb),
        .mem_write_valid(mem_write_valid), .mem_write_ready(mem_write_ready), .mem_write_data(mem_write_data), .mem_write_strb(mem_write_strb), .mem_write_last(mem_write_last),
        .mem_read_valid(mem_read_valid), .mem_read_ready(mem_read_ready), .mem_read_data(mem_read_data), .mem_read_last(mem_read_last)
    );
endmodule
