// Board-local AXI-Lite register bank. AW/W are captured independently before
// decode. Read selection is split into 8-word banks and a registered final mux.
// This does not alter the shared DeiT/GEMM control module.
module llm_sparse_board_control (
    input logic aclk,aresetn,
    input logic [7:0] s_axi_awaddr,
    input logic s_axi_awvalid, output logic s_axi_awready,
    input logic [31:0] s_axi_wdata,
    input logic [3:0] s_axi_wstrb,
    input logic s_axi_wvalid, output logic s_axi_wready,
    output logic [1:0] s_axi_bresp, output logic s_axi_bvalid,
    input logic s_axi_bready,
    input logic [7:0] s_axi_araddr,
    input logic s_axi_arvalid, output logic s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp, output logic s_axi_rvalid,
    input logic s_axi_rready,
    input logic [31:0] status_word,
    input logic [511:0] perf_regs_flat,
    output wire [31:0] ctrl_word,
    output wire [511:0] regs_flat
);
    logic [31:0] regs[0:15];
    wire [31:0] readable[0:31];
    logic aw_full,w_full;
    logic [5:0] aw_index,read_index;
    logic [31:0] write_data;
    logic [3:0] write_strobe;
    logic [1:0] read_state,read_bank;
    logic read_zero;
    logic [31:0] bank_data[0:3];
    assign ctrl_word=regs[0];
    for(genvar i=0;i<16;i++)begin
        assign regs_flat[i*32+:32]=i==1 ? status_word : regs[i];
        assign readable[i]=i==1 ? status_word : regs[i];
        assign readable[i+16]=perf_regs_flat[i*32+:32];
    end
    assign s_axi_awready=!aw_full && !s_axi_bvalid;
    assign s_axi_wready=!w_full && !s_axi_bvalid;
    assign s_axi_arready=read_state==0;
    assign s_axi_bresp=0;
    assign s_axi_rresp=0;
    always_ff @(posedge aclk)begin
        if(!aresetn)begin
            aw_full<=0;w_full<=0;s_axi_bvalid<=0;s_axi_rvalid<=0;
            read_state<=0;s_axi_rdata<=0;
            aw_index<=0;write_data<=0;write_strobe<=0;read_index<=0;
            read_bank<=0;read_zero<=0;
            for(integer i=0;i<16;i++)regs[i]<=0;
        end else begin
            if(s_axi_awvalid && s_axi_awready)begin aw_full<=1;aw_index<=s_axi_awaddr[7:2];end
            if(s_axi_wvalid && s_axi_wready)begin w_full<=1;write_data<=s_axi_wdata;write_strobe<=s_axi_wstrb;end
            if(aw_full && w_full && !s_axi_bvalid)begin
                if(aw_index<16 && aw_index!=1)
                    for(integer b=0;b<4;b++)if(write_strobe[b])regs[aw_index[3:0]][b*8+:8]<=write_data[b*8+:8];
                aw_full<=0;w_full<=0;s_axi_bvalid<=1;
            end else if(s_axi_bvalid && s_axi_bready)s_axi_bvalid<=0;
            case(read_state)
                0:if(s_axi_arvalid && s_axi_arready)begin read_index<=s_axi_araddr[7:2];read_state<=1;end
                1:begin
                    for(integer b=0;b<4;b++)bank_data[b]<=readable[b*8+read_index[2:0]];
                    read_bank<=read_index[4:3];read_zero<=read_index[5];read_state<=2;
                end
                2:begin s_axi_rdata<=read_zero ? 0 : bank_data[read_bank];s_axi_rvalid<=1;read_state<=3;end
                3:if(s_axi_rvalid && s_axi_rready)begin s_axi_rvalid<=0;read_state<=0;end
            endcase
        end
    end
endmodule
