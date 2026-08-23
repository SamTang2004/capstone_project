module tb_bcuda_transformer_ddr_top;
    logic clk=0,rst_n=0; always #5 clk=~clk;
    logic [7:0] awaddr,araddr;logic awvalid,awready,wvalid,wready,bvalid,bready,arvalid,arready,rvalid,rready;
    logic [31:0] wdata,rdata;logic [3:0] wstrb;logic [1:0] bresp,rresp;
    logic mem_cmd_valid,mem_cmd_ready,mem_cmd_write;logic [31:0] mem_cmd_addr,mem_cmd_wdata;logic [3:0] mem_cmd_len,mem_cmd_wstrb;
    logic mem_write_valid,mem_write_ready=1,mem_write_last;logic [31:0] mem_write_data;logic [3:0] mem_write_strb;
    logic mem_read_valid,mem_read_ready,mem_read_last;logic [63:0] mem_read_data;
    logic [3:0] mem_debug_arlen=0;logic [2:0] mem_debug_arsize=3;logic [4:0] mem_debug_rbeats=0;logic [1:0] mem_debug_rresp=0;
    logic [7:0] mem[0:4095];logic read_active,write_active;logic [31:0] physical_addr;logic [3:0] beat,write_word;
    integer errors=0;

    bcuda_transformer_ddr_top dut(.aclk(clk),.aresetn(rst_n),.s_axi_awaddr(awaddr),.s_axi_awvalid(awvalid),.s_axi_awready(awready),
        .s_axi_wdata(wdata),.s_axi_wstrb(wstrb),.s_axi_wvalid(wvalid),.s_axi_wready(wready),.s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_araddr(araddr),.s_axi_arvalid(arvalid),.s_axi_arready(arready),.s_axi_rdata(rdata),.s_axi_rresp(rresp),.s_axi_rvalid(rvalid),.s_axi_rready(rready),
        .mem_cmd_valid,.mem_cmd_ready,.mem_cmd_write,.mem_cmd_addr,.mem_cmd_len,.mem_cmd_wdata,.mem_cmd_wstrb,
        .mem_write_valid,.mem_write_ready,.mem_write_data,.mem_write_strb,.mem_write_last,
        .mem_read_valid,.mem_read_ready,.mem_read_data,.mem_read_last,
        .mem_debug_arlen,.mem_debug_arsize,.mem_debug_rbeats,.mem_debug_rresp);
    assign mem_cmd_ready=!read_active&&!write_active;
    assign mem_read_valid=read_active;
    assign mem_read_last=read_active&&beat==7;
    always_comb for(int i=0;i<8;i++)mem_read_data[i*8+:8]=mem[physical_addr+beat*8+i];
    always_ff @(posedge clk)begin
        if(!rst_n)begin read_active<=0;write_active<=0;physical_addr<=0;beat<=0;write_word<=0;end else begin
            if(mem_cmd_valid&&mem_cmd_ready)begin physical_addr<=mem_cmd_addr;beat<=0;write_word<=0;if(mem_cmd_write)write_active<=1;else read_active<=1;end
            if(mem_read_valid&&mem_read_ready)begin if(mem_read_last)read_active<=0;else beat<=beat+1;end
            if(mem_write_valid&&mem_write_ready)begin
                for(int i=0;i<4;i++)if(mem_write_strb[i])mem[physical_addr+write_word*4+i]<=mem_write_data[i*8+:8];
                if(mem_write_last)write_active<=0;else write_word<=write_word+1;
            end
        end
    end
    task automatic axi_write(input logic[7:0]addr,input logic[31:0]data);begin
        @(negedge clk);awaddr=addr;wdata=data;wstrb=4'hf;awvalid=1;wvalid=1;bready=1;
        while(!bvalid)@(posedge clk);@(negedge clk);awvalid=0;wvalid=0;bready=0;
    end endtask
    task automatic axi_read(input logic[7:0]addr,output logic[31:0]data);begin
        @(negedge clk);araddr=addr;arvalid=1;rready=1;while(!rvalid)@(negedge clk);data=rdata;
        @(posedge clk);@(negedge clk);arvalid=0;rready=0;
    end endtask
    task automatic put_qword(input int addr,input logic[63:0]value);
        begin for(int i=0;i<8;i++)mem[addr+i]=value[i*8+:8]; end
    endtask
    task automatic expect_i32(input int addr,input integer wanted);
        logic signed[31:0]got;
        begin
            got={mem[addr+3],mem[addr+2],mem[addr+1],mem[addr]};
            if(got!==wanted) begin
                $error("addr=%h expected=%0d got=%0d",addr,wanted,got);
                errors++;
            end
        end
    endtask
    task automatic put_i32(input int addr,input integer value);
        begin for(int i=0;i<4;i++)mem[addr+i]=value>>(i*8); end
    endtask
    initial begin
        logic[31:0]status;
        for(int i=0;i<4096;i++)mem[i]=0;
        awaddr=0;awvalid=0;wdata=0;wstrb=0;wvalid=0;bready=0;araddr=0;arvalid=0;rready=0;
        put_qword('h100,64'h0000000000050301);put_qword('h108,64'h0000000000060402);
        put_qword('h400,64'h0000000000000807);put_qword('h408,64'h0000000000000a09);
        repeat(12)@(posedge clk);rst_n=1;
        axi_write('h08,'h100);axi_write('h0c,'h400);axi_write('h10,'h800);axi_write('h1c,8);
        axi_write('h20,3);axi_write('h24,2);axi_write('h28,2);axi_write('h2c,8);axi_write('h30,1);
        axi_write('h00,1);
        status=0;for(int poll=0;poll<2000&&!(status&2);poll++)axi_read('h04,status);
        if(!(status&2)||status&4) begin $error("top status=%h",status);errors++;end
        expect_i32('h800,25);expect_i32('h804,28);expect_i32('h808,57);expect_i32('h80c,64);expect_i32('h810,89);expect_i32('h814,100);

        // Legacy row-major requantize must route to requantize_ddr rather than
        // falling through to the GEMM scheduler with stale dimensions.
        axi_write('h00,0);
        put_i32('hb00,1);put_i32('hb04,-2);put_i32('hb08,200);
        mem['hc00]=8'haa;mem['hc01]=8'haa;mem['hc02]=8'haa;mem['hc03]=8'h5a;mem['hc40]=8'ha5;
        axi_write('h08,'hb00);axi_write('h10,'hc00);axi_write('h20,3);
        axi_write('h34,1);axi_write('h38,0);axi_write('h3c,0);axi_write('h00,3);
        status=0;for(int poll=0;poll<4000&&!(status&2);poll++)axi_read('h04,status);
        if(!(status&2)||status&4) begin $error("requant status=%h",status);errors++;end
        if({mem['hc02],mem['hc01],mem['hc00]}!==24'h7ffe01)begin $error("row-major requant failed");errors++;end
        if(mem['hc03]!==8'h5a||mem['hc40]!==8'ha5)begin $error("row-major requant overrun");errors++;end

        // Fused row-major INT32 to tile-major INT8 requantize/pack operation.
        axi_write('h00,0);
        put_i32('h900,1);put_i32('h904,-2);put_i32('h908,200);
        put_i32('h90c,4);put_i32('h910,5);put_i32('h914,6);
        put_i32('h918,-200);put_i32('h91c,8);put_i32('h920,9);
        axi_write('h08,'h900);axi_write('h10,'ha00);axi_write('h14,12);
        axi_write('h20,3);axi_write('h24,3);axi_write('h34,1);axi_write('h38,0);axi_write('h3c,0);
        axi_write('h00,'h1001);
        status=0;for(int poll=0;poll<4000&&!(status&2);poll++)axi_read('h04,status);
        if(!(status&2)||status&4) begin $error("pack status=%h",status);errors++;end
        if({mem['ha02],mem['ha01],mem['ha00]}!==24'h800401)begin $error("pack column0 failed");errors++;end
        if({mem['ha0a],mem['ha09],mem['ha08]}!==24'h0805fe)begin $error("pack column1 failed");errors++;end
        if({mem['ha12],mem['ha11],mem['ha10]}!==24'h09067f)begin $error("pack column2 failed");errors++;end
        if(errors)$fatal(1,"Integrated Transformer top failed errors=%0d",errors);
        $display("Integrated Transformer DDR top golden trace passed");$finish;
    end
    initial begin#1000000;$fatal(1,"Integrated top timeout state=%0d",dut.scheduler.state);end
endmodule
