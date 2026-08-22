module tb_memory_io_axi3_bridge_burst;
    logic [3:0] debug_arlen;
    logic [2:0] debug_arsize;
    logic [4:0] debug_rbeats;
    logic [1:0] debug_rresp;
    logic clk=0, rst_n=0, cmd_valid, cmd_ready, cmd_write;
    logic [31:0] cmd_addr, cmd_wdata, write_data;
    logic [63:0] read_data;
    logic [3:0] cmd_len, cmd_wstrb, write_strb;
    logic write_valid, write_ready, write_last, read_valid, read_last, read_ready, error;
    logic [31:0] awaddr,araddr; logic [63:0] wdata,rdata; logic [7:0] wstrb; logic [3:0] awlen,arlen;
    logic [2:0] awsize,arsize; logic [1:0] awburst,arburst,bresp=0,rresp=0;
    logic [5:0] awid,wid,bid=0,arid,rid=0;
    logic [1:0] awlock,arlock;
    logic [3:0] awcache,awqos,arcache,arqos;
    logic [2:0] awprot,arprot;
    logic awvalid,awready=1,wlast,wvalid,wready=1,bvalid=0,bready;
    logic arvalid,arready=1,rvalid=0,rlast,rready;
    logic [7:0] mem[0:255]; integer wi,ri,errors=0;
    always #5 clk=~clk;
    initial begin repeat(5000)@(posedge clk);$fatal(1,"AXI burst bridge timeout state=%0d",dut.state);end

    memory_io_axi3_bridge dut(.*,
        .aclk(clk),.aresetn(rst_n),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),
        .m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),
        .m_axi_awcache(awcache),.m_axi_awprot(awprot),.m_axi_awqos(awqos),
        .m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_awid(awid),
        .m_axi_wid(wid),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),
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
            if(wvalid&&wready) begin
                for (int lane=0;lane<8;lane++)
                    if(wstrb[lane]) mem[awaddr+wi*8+lane]<=wdata[lane*8+:8];
                if(wlast)bvalid<=1; else wi<=wi+1;
            end
            if(bvalid&&bready) begin bvalid<=0;wi<=0; end
            if(arvalid&&arready) begin
                ri<=0;
                for(int lane=0;lane<8;lane++) rdata[lane*8+:8]<=mem[araddr+lane];
                rvalid<=1;rlast<=(arlen==0);
            end
            else if(rvalid&&rready) begin
                if(rlast) begin rvalid<=0;rlast<=0; end
                else begin
                    ri<=ri+1;
                    for(int lane=0;lane<8;lane++) rdata[lane*8+:8]<=mem[araddr+(ri+1)*8+lane];
                    rlast<=(ri+1==arlen);
                end
            end
        end
    end

    initial begin
        cmd_valid=0;cmd_write=0;cmd_addr=0;cmd_len=0;cmd_wdata=0;cmd_wstrb=0;
        write_valid=0;write_data=0;write_strb=4'hf;write_last=0;read_ready=0;
        repeat(3)@(posedge clk);rst_n=1;
        @(negedge clk);cmd_write=1;cmd_addr=32'h20;cmd_len=3;cmd_valid=1;
        @(negedge clk);cmd_valid=0; wait(write_ready);
        for(int i=0;i<4;i++) begin
            @(negedge clk);write_valid=1;write_data=32'h100+i;write_last=(i==3);
            do @(posedge clk); while(!write_ready);
        end
        @(negedge clk);write_valid=0;write_last=0;wait(cmd_ready);
        if(awid!=0||wid!=0||awlen!=1||awsize!=3||awburst!=1)errors++;

        // A burst beginning at address +4 starts in the upper bus lanes and
        // alternates lanes as each 32-bit narrow transfer advances.
        @(negedge clk);cmd_write=1;cmd_addr=32'h44;cmd_len=2;cmd_valid=1;
        @(negedge clk);cmd_valid=0;wait(write_ready);
        for(int i=0;i<3;i++) begin
            @(negedge clk);write_valid=1;write_data=32'h200+i;write_last=(i==2);
            do @(posedge clk); while(!write_ready);
        end
        @(negedge clk);write_valid=0;write_last=0;wait(cmd_ready);
        if(awaddr!=32'h40||awlen!=1||awsize!=3 ||
           {mem[71],mem[70],mem[69],mem[68]}!=32'h200 ||
           {mem[75],mem[74],mem[73],mem[72]}!=32'h201 ||
           {mem[79],mem[78],mem[77],mem[76]}!=32'h202) errors++;

        // Maximum logical burst from an upper lane occupies nine physical
        // beats. This guards the widened burst-length addition from wrapping.
        @(negedge clk);cmd_write=1;cmd_addr=32'h84;cmd_len=15;cmd_valid=1;
        @(negedge clk);cmd_valid=0;wait(write_ready);
        for(int i=0;i<16;i++) begin
            @(negedge clk);write_valid=1;write_data=32'h300+i;write_last=(i==15);
            do @(posedge clk); while(!write_ready);
        end
        @(negedge clk);write_valid=0;write_last=0;wait(cmd_ready);
        if(awaddr!=32'h80||awlen!=8||awsize!=3 ||
           {mem[135],mem[134],mem[133],mem[132]}!=32'h300 ||
           {mem[195],mem[194],mem[193],mem[192]}!=32'h30f) errors++;

        @(negedge clk);cmd_write=0;cmd_addr=32'h20;cmd_len=1;cmd_valid=1;
        @(negedge clk);cmd_valid=0;read_ready=1;
        for(int i=0;i<2;i++) begin
            wait(read_valid);#1;
            if(read_data!=={32'h100+i*2+1,32'h100+i*2}||read_last!=(i==1))errors++;
            @(posedge clk);
        end
        if(arid!=0||arlen!=1||arsize!=3||arburst!=1||error||errors) begin
            $display("bridge diag awaddr=%h awlen=%0d awsize=%0d mem=%h,%h,%h,%h arlen=%0d errors=%0d",
                awaddr,awlen,awsize,{mem[35],mem[34],mem[33],mem[32]},
                {mem[39],mem[38],mem[37],mem[36]},
                {mem[43],mem[42],mem[41],mem[40]},
                {mem[47],mem[46],mem[45],mem[44]},arlen,errors);
            $fatal(1,"AXI3 burst bridge failed errors=%0d",errors);
        end
        $display("AXI3 burst bridge golden trace passed: logical words packed into native 64-bit beats");$finish;
    end
endmodule
