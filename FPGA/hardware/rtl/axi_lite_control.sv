module axi_lite_control #(
    parameter int ADDR_WIDTH = 7
) (
    input  logic                  aclk,
    input  logic                  aresetn,

    input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                  s_axi_awvalid,
    output logic                  s_axi_awready,
    input  logic [31:0]           s_axi_wdata,
    input  logic [3:0]            s_axi_wstrb,
    input  logic                  s_axi_wvalid,
    output logic                  s_axi_wready,
    output logic [1:0]            s_axi_bresp,
    output logic                  s_axi_bvalid,
    input  logic                  s_axi_bready,

    input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                  s_axi_arvalid,
    output logic                  s_axi_arready,
    output logic [31:0]           s_axi_rdata,
    output logic [1:0]            s_axi_rresp,
    output logic                  s_axi_rvalid,
    input  logic                  s_axi_rready,

    input  logic [31:0]           status_word,
    input  logic [32*16-1:0]      diagnostics_flat,
    output logic [31:0]           ctrl_word,
    output logic [32*16-1:0]      regs_flat
);

    logic [31:0] regs [0:15];
    logic [3:0] read_index;
    logic [3:0] diagnostic_read_index;
    logic                    aw_holding;
    logic                    w_holding;
    logic [ADDR_WIDTH-1:0]   awaddr_holding;
    logic [31:0]             wdata_holding;
    logic [3:0]              wstrb_holding;
    logic [3:0]              write_index_selected;
    logic                    write_extended_selected;
    logic [31:0]             write_data_selected;
    logic [3:0]              write_strb_selected;

    assign read_index = s_axi_araddr[5:2];
    assign diagnostic_read_index = s_axi_araddr[5:2];
    assign ctrl_word = regs[0];
    // AXI-Lite permits AW and W to arrive in different cycles.  Hold each
    // channel until its peer arrives, then produce exactly one write response.
    assign s_axi_awready = !aw_holding && !s_axi_bvalid;
    assign s_axi_wready = !w_holding && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    assign write_index_selected = aw_holding ? awaddr_holding[5:2] : s_axi_awaddr[5:2];
    assign write_extended_selected =
        aw_holding ? awaddr_holding[6] : s_axi_awaddr[6];
    assign write_data_selected = w_holding ? wdata_holding : s_axi_wdata;
    assign write_strb_selected = w_holding ? wstrb_holding : s_axi_wstrb;

    genvar reg_idx;
    generate
        for (reg_idx = 0; reg_idx < 16; reg_idx = reg_idx + 1) begin : flatten_regs
            assign regs_flat[reg_idx*32 +: 32] = regs[reg_idx];
        end
    endgenerate

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_rvalid <= 1'b0;
            s_axi_rresp <= 2'b00;
            s_axi_rdata <= '0;
            aw_holding <= 1'b0;
            w_holding <= 1'b0;
            awaddr_holding <= '0;
            wdata_holding <= '0;
            wstrb_holding <= '0;
            for (int i = 0; i < 16; i++) begin
                regs[i] <= '0;
            end
        end else begin
            regs[1] <= status_word;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_holding <= 1'b1;
                awaddr_holding <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_holding <= 1'b1;
                wdata_holding <= s_axi_wdata;
                wstrb_holding <= s_axi_wstrb;
            end

            if (!s_axi_bvalid &&
                (aw_holding || (s_axi_awvalid && s_axi_awready)) &&
                (w_holding || (s_axi_wvalid && s_axi_wready))) begin
                if (!write_extended_selected) begin
                    for (int b = 0; b < 4; b++) begin
                        if (write_strb_selected[b]) begin
                            regs[write_index_selected][8*b +: 8] <=
                                write_data_selected[8*b +: 8];
                        end
                    end
                end
                aw_holding <= 1'b0;
                w_holding <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rdata <= s_axi_araddr[6] ?
                    diagnostics_flat[diagnostic_read_index*32 +: 32] :
                    ((read_index == 4'd1) ? status_word : regs[read_index]);
                s_axi_rvalid <= 1'b1;
                s_axi_rresp <= 2'b00;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
