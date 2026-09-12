module llm_axi_queued_ip (
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET aresetn" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *) input wire aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *) input wire aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input wire [7:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output wire [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output wire s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [7:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output wire [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output wire s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire s_axi_rready,
    output wire mem_cmd_valid,input wire mem_cmd_ready,output wire mem_cmd_write,output wire [31:0] mem_cmd_addr,output wire [3:0] mem_cmd_len,
    output wire [31:0] mem_cmd_wdata,output wire [3:0] mem_cmd_wstrb,output wire mem_write_valid,input wire mem_write_ready,
    output wire [31:0] mem_write_data,output wire [3:0] mem_write_strb,output wire mem_write_last,
    input wire mem_read_valid,output wire mem_read_ready,input wire [63:0] mem_read_data,input wire mem_read_last,
    input wire [3:0] mem_debug_arlen,input wire [2:0] mem_debug_arsize,input wire [4:0] mem_debug_rbeats,input wire [1:0] mem_debug_rresp
);
    llm_axi_queued_top accelerator (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .mem_cmd_valid(mem_cmd_valid), .mem_cmd_ready(mem_cmd_ready), .mem_cmd_write(mem_cmd_write),
        .mem_cmd_addr(mem_cmd_addr), .mem_cmd_len(mem_cmd_len), .mem_cmd_wdata(mem_cmd_wdata), .mem_cmd_wstrb(mem_cmd_wstrb),
        .mem_write_valid(mem_write_valid), .mem_write_ready(mem_write_ready), .mem_write_data(mem_write_data),
        .mem_write_strb(mem_write_strb), .mem_write_last(mem_write_last),
        .mem_read_valid(mem_read_valid), .mem_read_ready(mem_read_ready), .mem_read_data(mem_read_data), .mem_read_last(mem_read_last),
        .mem_debug_arlen(mem_debug_arlen), .mem_debug_arsize(mem_debug_arsize),
        .mem_debug_rbeats(mem_debug_rbeats), .mem_debug_rresp(mem_debug_rresp)
    );
endmodule
