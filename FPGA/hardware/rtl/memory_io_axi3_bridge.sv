// Native AXI3 bridge for Zynq-7000 HP ports. Keeping the PL master AXI3
// avoids an AXI4-to-AXI3 protocol converter in the latency-critical DDR path.
module memory_io_axi3_bridge (
    input  logic        aclk,
    input  logic        aresetn,
    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic        cmd_write,
    input  logic [31:0] cmd_addr,
    input  logic [3:0]  cmd_len,
    input  logic [31:0] cmd_wdata,
    input  logic [3:0]  cmd_wstrb,
    input  logic        write_valid,
    output logic        write_ready,
    input  logic [31:0] write_data,
    input  logic [3:0]  write_strb,
    input  logic        write_last,
    output logic        read_valid,
    output logic        read_last,
    input  logic        read_ready,
    output logic [31:0] read_data,
    output logic        error,

    output logic [5:0]  m_axi_awid,
    output logic [31:0] m_axi_awaddr,
    output logic [3:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic [1:0]  m_axi_awlock,
    output logic [3:0]  m_axi_awcache,
    output logic [2:0]  m_axi_awprot,
    output logic [3:0]  m_axi_awqos,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [5:0]  m_axi_wid,
    output logic [31:0] m_axi_wdata,
    output logic [3:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [5:0]  m_axi_bid,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,
    output logic [5:0]  m_axi_arid,
    output logic [31:0] m_axi_araddr,
    output logic [3:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic [1:0]  m_axi_arlock,
    output logic [3:0]  m_axi_arcache,
    output logic [2:0]  m_axi_arprot,
    output logic [3:0]  m_axi_arqos,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [5:0]  m_axi_rid,
    input  logic [31:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rvalid,
    input  logic        m_axi_rlast,
    output logic        m_axi_rready
);
    typedef enum logic [2:0] {IDLE, WRITE_ADDR, WRITE_DATA, WRITE_RESP, READ_ADDR, READ_DATA} state_t;
    state_t state;
    logic [31:0] addr_r;
    logic [3:0] len_r;

    assign cmd_ready = state == IDLE;
    assign m_axi_awid = 6'd0;
    assign m_axi_awaddr = addr_r;
    assign m_axi_awlen = len_r;
    assign m_axi_awsize = 3'd2;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awlock = 2'b00;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot = 3'b000;
    assign m_axi_awqos = 4'd0;
    assign m_axi_awvalid = state == WRITE_ADDR;
    assign m_axi_wid = 6'd0;
    assign m_axi_wdata = write_data;
    assign m_axi_wstrb = write_strb;
    assign m_axi_wlast = write_last;
    assign m_axi_wvalid = state == WRITE_DATA && write_valid;
    assign write_ready = state == WRITE_DATA && m_axi_wready;
    assign m_axi_bready = state == WRITE_RESP;
    assign m_axi_arid = 6'd0;
    assign m_axi_araddr = addr_r;
    assign m_axi_arlen = len_r;
    assign m_axi_arsize = 3'd2;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock = 2'b00;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot = 3'b000;
    assign m_axi_arqos = 4'd0;
    assign m_axi_arvalid = state == READ_ADDR;
    assign read_valid = state == READ_DATA && m_axi_rvalid;
    assign read_last = read_valid && m_axi_rlast;
    assign read_data = m_axi_rdata;
    assign m_axi_rready = state == READ_DATA && read_ready;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state <= IDLE;
            addr_r <= 32'd0;
            len_r <= 4'd0;
            error <= 1'b0;
        end else begin
            if (cmd_valid && cmd_ready) begin
                addr_r <= cmd_addr;
                len_r <= cmd_len;
                state <= cmd_write ? WRITE_ADDR : READ_ADDR;
            end
            unique case (state)
                IDLE: begin end
                WRITE_ADDR: if (m_axi_awready) state <= WRITE_DATA;
                WRITE_DATA: if (write_valid && m_axi_wready && write_last) state <= WRITE_RESP;
                WRITE_RESP: if (m_axi_bvalid) begin
                    error <= error || m_axi_bresp != 2'b00 || m_axi_bid != 6'd0;
                    state <= IDLE;
                end
                READ_ADDR: if (m_axi_arready) state <= READ_DATA;
                READ_DATA: if (m_axi_rvalid && read_ready) begin
                    error <= error || m_axi_rresp != 2'b00 || m_axi_rid != 6'd0;
                    if (m_axi_rlast) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
