`timescale 1ns/1ps
// Bridges the accelerator command protocol to AXI4. Read commands are queued
// and issued independently of returned data, permitting several same-ID AXI
// bursts to be outstanding. AXI preserves response order for a common ID.
module memory_io_axi_bridge #(
    parameter int READ_CMD_DEPTH = 8
) (
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

    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [31:0] m_axi_wdata,
    output logic [3:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [31:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rvalid,
    input  logic        m_axi_rlast,
    output logic        m_axi_rready
);
    localparam int READ_PTR_W = $clog2(READ_CMD_DEPTH);

    typedef enum logic [1:0] {
        WRITE_IDLE, WRITE_ADDR, WRITE_DATA, WRITE_RESP
    } write_state_t;

    write_state_t write_state;
    logic [31:0] write_addr_r;
    logic [3:0] write_len_r;
    logic [31:0] read_addr_fifo [0:READ_CMD_DEPTH-1];
    logic [3:0] read_len_fifo [0:READ_CMD_DEPTH-1];
    logic [READ_PTR_W-1:0] read_head_r, read_tail_r;
    logic [READ_PTR_W:0] read_count_r;
    logic [READ_PTR_W:0] read_outstanding_r;

    wire accept_read = cmd_valid && cmd_ready && !cmd_write;
    wire accept_write = cmd_valid && cmd_ready && cmd_write;
    wire issue_read = m_axi_arvalid && m_axi_arready;
    wire finish_read = m_axi_rvalid && m_axi_rready && m_axi_rlast;

    assign cmd_ready = cmd_write ?
        (write_state == WRITE_IDLE && read_count_r == 0 &&
         read_outstanding_r == 0) :
        (write_state == WRITE_IDLE && read_count_r < READ_CMD_DEPTH);

    assign m_axi_awaddr = write_addr_r;
    assign m_axi_awlen = {4'd0, write_len_r};
    assign m_axi_awsize = 3'd2;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = write_state == WRITE_ADDR;
    assign m_axi_wdata = write_data;
    assign m_axi_wstrb = write_strb;
    assign m_axi_wlast = write_last;
    assign m_axi_wvalid = write_state == WRITE_DATA && write_valid;
    assign write_ready = write_state == WRITE_DATA && m_axi_wready;
    assign m_axi_bready = write_state == WRITE_RESP;

    assign m_axi_araddr = read_addr_fifo[read_head_r];
    assign m_axi_arlen = {4'd0, read_len_fifo[read_head_r]};
    assign m_axi_arsize = 3'd2;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = read_count_r != 0 &&
                           write_state == WRITE_IDLE;
    assign read_valid = m_axi_rvalid;
    assign read_last = m_axi_rvalid && m_axi_rlast;
    assign read_data = m_axi_rdata;
    assign m_axi_rready = read_ready;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            write_state <= WRITE_IDLE;
            write_addr_r <= '0;
            write_len_r <= '0;
            read_head_r <= '0;
            read_tail_r <= '0;
            read_count_r <= '0;
            read_outstanding_r <= '0;
            error <= 1'b0;
        end else begin
            if (accept_read) begin
                read_addr_fifo[read_tail_r] <= cmd_addr;
                read_len_fifo[read_tail_r] <= cmd_len;
                read_tail_r <= read_tail_r + 1'b1;
            end
            if (issue_read)
                read_head_r <= read_head_r + 1'b1;

            unique case ({accept_read, issue_read})
                2'b10: read_count_r <= read_count_r + 1'b1;
                2'b01: read_count_r <= read_count_r - 1'b1;
                default: read_count_r <= read_count_r;
            endcase
            unique case ({issue_read, finish_read})
                2'b10: read_outstanding_r <= read_outstanding_r + 1'b1;
                2'b01: read_outstanding_r <= read_outstanding_r - 1'b1;
                default: read_outstanding_r <= read_outstanding_r;
            endcase

            if (m_axi_rvalid && m_axi_rready &&
                m_axi_rresp != 2'b00)
                error <= 1'b1;

            unique case (write_state)
                WRITE_IDLE: if (accept_write) begin
                    write_addr_r <= cmd_addr;
                    write_len_r <= cmd_len;
                    write_state <= WRITE_ADDR;
                end
                WRITE_ADDR: if (m_axi_awready)
                    write_state <= WRITE_DATA;
                WRITE_DATA: if (write_valid && m_axi_wready &&
                                write_last)
                    write_state <= WRITE_RESP;
                WRITE_RESP: if (m_axi_bvalid) begin
                    if (m_axi_bresp != 2'b00)
                        error <= 1'b1;
                    write_state <= WRITE_IDLE;
                end
                default: write_state <= WRITE_IDLE;
            endcase
        end
    end
endmodule
