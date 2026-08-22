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
    output logic [63:0] read_data,
    output logic        error,
    output logic [3:0]  debug_arlen,
    output logic [2:0]  debug_arsize,
    output logic [4:0]  debug_rbeats,
    output logic [1:0]  debug_rresp,

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
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
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
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rvalid,
    input  logic        m_axi_rlast,
    output logic        m_axi_rready
);
    typedef enum logic [2:0] {IDLE, WRITE_ADDR, WRITE_DATA, WRITE_RESP, READ_ADDR, READ_DATA} state_t;
    state_t state;
    logic [31:0] addr_r;
    logic [3:0] len_r;
    logic        write_have_lower;
    logic        write_flush_lower;
    logic [31:0] write_lower_data;
    logic [3:0]  write_lower_strb;

    assign cmd_ready = state == IDLE;
    assign m_axi_awid = 6'd0;
    assign m_axi_awaddr = {addr_r[31:3], 3'b000};
    // The producer counts 32-bit words while HP0 consumes native 64-bit
    // beats. Include an initial upper-word offset, then round up by two.
    assign m_axi_awlen = ({1'b0, len_r} + addr_r[2]) >> 1;
    assign m_axi_awsize = 3'd3;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awlock = 2'b00;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot = 3'b000;
    assign m_axi_awqos = 4'd0;
    assign m_axi_awvalid = state == WRITE_ADDR;
    assign m_axi_wid = 6'd0;
    // Pair logical words into native HP0 beats. A final unpaired lower word
    // is flushed with its upper strobes clear. For a +4 start, the seeded
    // empty lower half causes the first producer word to occupy upper lanes.
    assign m_axi_wdata = write_flush_lower ? {32'd0, write_lower_data} :
                         {write_data, write_lower_data};
    assign m_axi_wstrb = write_flush_lower ? {4'd0, write_lower_strb} :
                         {write_strb, write_lower_strb};
    assign m_axi_wlast = write_flush_lower || (write_valid && write_last);
    assign m_axi_wvalid = state == WRITE_DATA &&
                         (write_flush_lower || (write_have_lower && write_valid));
    assign write_ready = state == WRITE_DATA && !write_flush_lower &&
                         (!write_have_lower || m_axi_wready);
    assign m_axi_bready = state == WRITE_RESP;
    assign m_axi_arid = 6'd0;
    assign m_axi_araddr = addr_r;
    assign m_axi_arlen = len_r;
    assign m_axi_arsize = 3'd3;
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
            write_have_lower <= 1'b0;
            write_flush_lower <= 1'b0;
            write_lower_data <= 32'd0;
            write_lower_strb <= 4'd0;
            error <= 1'b0;
            debug_arlen <= 4'd0;
            debug_arsize <= 3'd0;
            debug_rbeats <= 5'd0;
            debug_rresp <= 2'd0;
        end else begin
            if (cmd_valid && cmd_ready) begin
                addr_r <= cmd_addr;
                len_r <= cmd_len;
                write_have_lower <= cmd_write && cmd_addr[2];
                write_flush_lower <= 1'b0;
                write_lower_data <= 32'd0;
                write_lower_strb <= 4'd0;
                state <= cmd_write ? WRITE_ADDR : READ_ADDR;
                if (!cmd_write) begin
                    debug_arlen <= cmd_len;
                    debug_arsize <= 3'd3;
                    debug_rbeats <= 5'd0;
                    debug_rresp <= 2'd0;
                end
            end
            unique case (state)
                IDLE: begin end
                WRITE_ADDR: if (m_axi_awready) state <= WRITE_DATA;
                WRITE_DATA: begin
                    if (write_flush_lower && m_axi_wready) begin
                        write_flush_lower <= 1'b0;
                        write_have_lower <= 1'b0;
                        state <= WRITE_RESP;
                    end else if (write_valid && write_ready) begin
                        if (!write_have_lower) begin
                            write_lower_data <= write_data;
                            write_lower_strb <= write_strb;
                            write_have_lower <= 1'b1;
                            if (write_last) write_flush_lower <= 1'b1;
                        end else begin
                            write_have_lower <= 1'b0;
                            if (write_last) state <= WRITE_RESP;
                        end
                    end
                end
                WRITE_RESP: if (m_axi_bvalid) begin
                    error <= error || m_axi_bresp != 2'b00 || m_axi_bid != 6'd0;
                    state <= IDLE;
                end
                READ_ADDR: if (m_axi_arready) state <= READ_DATA;
                READ_DATA: if (m_axi_rvalid && read_ready) begin
                    error <= error || m_axi_rresp != 2'b00 || m_axi_rid != 6'd0;
                    debug_rbeats <= debug_rbeats + 1'b1;
                    debug_rresp <= m_axi_rresp;
                    if (m_axi_rlast) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
