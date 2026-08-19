module tb_memory_io_axi3_bridge_burst;
    logic clk=0, rst_n=0, cmd_valid, cmd_ready, cmd_write;
    logic [31:0] cmd_addr, cmd_wdata, write_data, read_data;
    logic [3:0] cmd_len, cmd_wstrb, write_strb;
    logic write_valid, write_ready, write_last, read_valid, read_last, read_ready, error;
    logic [31:0] awaddr,wdata,araddr,rdata; logic [3:0] awlen,arlen;
    logic [2:0] awsize,arsize; logic [1:0] awburst,arburst,bresp=0,rresp=0;
    logic [5:0] awid,wid,bid=0,arid,rid=0;
    logic [1:0] awlock,arlock;
    logic [3:0] awcache,awqos,arcache,arqos;
    logic [2:0] awprot,arprot;
    logic awvalid,awready=1,wlast,wvalid,wready=1,bvalid=0,bready;
    logic arvalid,arready=1,rvalid=0,rlast,rready;
    logic [31:0] mem[0:31]; integer wi,ri,errors=0;
    always #5 clk=~clk;
    initial begin repeat(5000)@(posedge clk);$fatal(1,"AXI burst bridge timeout state=%0d",dut.state);end

    memory_io_axi3_bridge dut(.*,
        .aclk(clk),.aresetn(rst_n),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),
        .m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),
        .m_axi_awcache(awcache),.m_axi_awprot(awprot),.m_axi_awqos(awqos),
        .m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_awid(awid),
        .m_axi_wid(wid),.m_axi_wdata(wdata),.m_axi_wstrb(),.m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid),.m_axi_wready(wready),.m_axi_bid(bid),
        .m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
        .m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),
        .m_axi_arcache(arcache),.m_axi_arprot(arprot),.m_axi_arqos(arqos),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),
        .m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rvalid(rvalid),
        .m_axi_rlast(rlast),.m_axi_rready(rready));

    always_ff @(posedge clk) begin
        if(!rst_n) begin bvalid<=0;rvalid<=0;rlast<=0;wi<=0;ri<=0; end else begin
            if(wvalid&&wready) begin mem[(awaddr>>2)+wi]<=wdata; if(wlast)bvalid<=1; else wi<=wi+1; end
            if(bvalid&&bready) begin bvalid<=0;wi<=0; end
            if(arvalid&&arready) begin ri<=0;rdata<=mem[araddr>>2];rvalid<=1;rlast<=(arlen==0); end
            else if(rvalid&&rready) begin
                if(rlast) begin rvalid<=0;rlast<=0; end
                else begin ri<=ri+1;rdata<=mem[(araddr>>2)+ri+1];rlast<=(ri+1==arlen); end
            end
        end
    end

    initial begin
        cmd_valid=0;cmd_write=0;cmd_addr=0;cmd_len=0;cmd_wdata=0;cmd_wstrb=0;
        write_valid=0;write_data=0;write_strb=4'hf;write_last=0;read_ready=0;
        repeat(3)@(posedge clk);rst_n=1;
        @(negedge clk);cmd_write=1;cmd_addr=32'h20;cmd_len=3;cmd_valid=1;
        @(negedge clk);cmd_valid=0; wait(write_ready);
        for(int i=0;i<4;i++) begin @(negedge clk);write_valid=1;write_data=32'h100+i;write_last=(i==3); wait(write_ready); end
        @(negedge clk);write_valid=0;write_last=0;wait(cmd_ready);
        if(awid!=0||wid!=0||awlen!=3||awsize!=2||awburst!=1)errors++;
        @(negedge clk);cmd_write=0;cmd_addr=32'h20;cmd_len=3;cmd_valid=1;
        @(negedge clk);cmd_valid=0;read_ready=1;
        for(int i=0;i<4;i++) begin wait(read_valid);#1;if(read_data!==32'h100+i||read_last!=(i==3))errors++;@(posedge clk); end
        if(arid!=0||arlen!=3||arsize!=2||arburst!=1||error||errors)$fatal(1,"AXI3 burst bridge failed errors=%0d",errors);
        $display("AXI3 burst bridge golden trace passed: 4 write beats, 4 read beats");$finish;
    end
endmodule
