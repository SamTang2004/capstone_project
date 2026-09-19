`timescale 1ns/1ps
module tb_sparse_board_control;
    logic aclk=0;always #2.5 aclk=~aclk;
    logic aresetn=0;
    logic [7:0] s_axi_awaddr=0,s_axi_araddr=0;
    logic s_axi_awvalid=0,s_axi_wvalid=0,s_axi_bready=0,s_axi_arvalid=0,s_axi_rready=0;
    wire s_axi_awready,s_axi_wready,s_axi_bvalid,s_axi_arready,s_axi_rvalid;
    logic [31:0] s_axi_wdata=0,status_word=32'h12345678;
    logic [3:0] s_axi_wstrb=0;
    wire [1:0] s_axi_bresp,s_axi_rresp;
    wire [31:0] s_axi_rdata,ctrl_word;
    wire [511:0] regs_flat;
    logic [511:0] perf_regs_flat=0;
    logic [31:0] expected[0:15];
    integer reads=0,writes=0;
    llm_sparse_board_control dut(.*);
    task automatic tick;
        @(posedge aclk);#1;@(negedge aclk);
    endtask
    task automatic wr(input integer addr,input logic[31:0] data,input logic[3:0] strobe,input integer ad,wd);
        bit sent_a,sent_w,hit_a,hit_w;
        integer t;
        sent_a=0;sent_w=0;t=0;s_axi_bready=0;
        s_axi_awaddr=8'(addr);s_axi_wdata=data;s_axi_wstrb=strobe;
        while(!s_axi_bvalid)begin
            s_axi_awvalid=!sent_a && t>=ad;s_axi_wvalid=!sent_w && t>=wd;
            hit_a=s_axi_awvalid && s_axi_awready;hit_w=s_axi_wvalid && s_axi_wready;
            tick();if(hit_a)sent_a=1;if(hit_w)sent_w=1;
            t++;if(t>100)$fatal(1,"AXI write timeout");
        end
        s_axi_awvalid=0;s_axi_wvalid=0;
        if(!sent_a || !sent_w || s_axi_bresp!=0)$fatal(1,"write response before AW/W");
        repeat(5)begin tick();if(!s_axi_bvalid)$fatal(1,"lost B under stall");end
        s_axi_bready=1;tick();s_axi_bready=0;writes++;
    endtask
    task automatic rd(input integer addr,input logic[31:0] value);
        integer t;
        s_axi_araddr=8'(addr);s_axi_arvalid=1;s_axi_rready=0;t=0;
        while(!s_axi_arready)tick();tick();s_axi_arvalid=0;
        while(!s_axi_rvalid)begin tick();t++;if(t>100)$fatal(1,"AXI read timeout");end
        if(s_axi_rdata!==value || s_axi_rresp!=0)$fatal(1,"read %h got=%h expected=%h",addr,s_axi_rdata,value);
        s_axi_araddr=8'hfc;
        repeat(5)begin tick();if(!s_axi_rvalid || s_axi_rdata!==value)$fatal(1,"lost R under stall");end
        s_axi_rready=1;tick();s_axi_rready=0;reads++;
    endtask
    initial begin
        repeat(5)tick();aresetn=1;tick();
        for(integer i=0;i<16;i++)begin
            expected[i]=0;perf_regs_flat[i*32+:32]=32'hffff0000+i;
        end
        for(integer i=0;i<16;i++)if(i!=1)begin
            expected[i]=32'h91827364+i;
            wr(i*4,expected[i],15,i%4,(i+2)%4);rd(i*4,expected[i]);
            wr(i*4,32'habcdef12,4'b0101,(i+1)%4,i%4);
            expected[i]=(expected[i]&32'hff00ff00)|32'h00cd0012;
            rd(i*4,expected[i]);
            if(regs_flat[i*32+:32]!==expected[i])$fatal(1,"staging mismatch");
        end
        wr(4,0,15,0,4);rd(4,status_word);
        for(integer i=0;i<16;i++)rd(64+i*4,32'hffff0000+i);
        wr(128,32'hdeadbeef,15,4,0);rd(128,0);rd(252,0);
        // Reset after only AW has arrived; no stale transaction may commit.
        s_axi_awaddr=8;s_axi_awvalid=1;tick();s_axi_awvalid=0;
        aresetn=0;repeat(2)tick();aresetn=1;tick();rd(8,0);
        wr(8,32'habcdef01,15,0,0);rd(8,32'habcdef01);
        $display("PASS: sparse board AXI writes=%0d reads=%0d",writes,reads);$finish;
    end
    initial begin #100000;$fatal(1,"test timeout");end
endmodule
