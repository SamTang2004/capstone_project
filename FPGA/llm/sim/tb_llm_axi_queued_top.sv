module tb_llm_axi_queued_top;
    logic clk=0,rst_n=0; always #5 clk=~clk;
    logic [7:0] awaddr,araddr;logic awvalid,awready,wvalid,wready,bvalid,bready,arvalid,arready,rvalid,rready;
    logic [31:0] wdata,rdata;logic [3:0] wstrb;logic [1:0] bresp,rresp;
    logic mem_cmd_valid,mem_cmd_ready,mem_cmd_write;logic [31:0] mem_cmd_addr,mem_cmd_wdata;logic [3:0] mem_cmd_len,mem_cmd_wstrb;
    logic mem_write_valid,mem_write_ready,mem_write_last;logic [31:0] mem_write_data;logic [3:0] mem_write_strb;
    logic mem_read_valid,mem_read_ready,mem_read_last;logic [63:0] mem_read_data;
    logic [3:0] mem_debug_arlen=0;logic [2:0] mem_debug_arsize=3;logic [4:0] mem_debug_rbeats=0;logic [1:0] mem_debug_rresp=0;
    logic [7:0] mem[0:4095];logic read_active,write_active;logic [31:0] physical_addr;logic [3:0] beat,write_word;
    integer errors=0;
    integer ticks=0;
    always @(posedge clk) ticks<=ticks+1;

    llm_axi_queued_top dut(.aclk(clk),.aresetn(rst_n),.s_axi_awaddr(awaddr),.s_axi_awvalid(awvalid),.s_axi_awready(awready),
        .s_axi_wdata(wdata),.s_axi_wstrb(wstrb),.s_axi_wvalid(wvalid),.s_axi_wready(wready),.s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_araddr(araddr),.s_axi_arvalid(arvalid),.s_axi_arready(arready),.s_axi_rdata(rdata),.s_axi_rresp(rresp),.s_axi_rvalid(rvalid),.s_axi_rready(rready),
        .mem_cmd_valid,.mem_cmd_ready,.mem_cmd_write,.mem_cmd_addr,.mem_cmd_len,.mem_cmd_wdata,.mem_cmd_wstrb,
        .mem_write_valid,.mem_write_ready,.mem_write_data,.mem_write_strb,.mem_write_last,
        .mem_read_valid,.mem_read_ready,.mem_read_data,.mem_read_last,
        .mem_debug_arlen,.mem_debug_arsize,.mem_debug_rbeats,.mem_debug_rresp);
    assign mem_cmd_ready=!read_active&&!write_active && ticks%3!=0;
    assign mem_read_valid=read_active && ticks%4!=0;
    assign mem_write_ready=ticks%3!=0;
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
        // AW and W deliberately arrive separately; B is backpressured.
        @(negedge clk);awaddr=addr;awvalid=1;bready=0;
        do @(posedge clk); while(!awready);
        @(negedge clk);awvalid=0;
        repeat(2)@(negedge clk);
        wdata=data;wstrb=4'hf;wvalid=1;
        do @(posedge clk); while(!wready);
        @(negedge clk);wvalid=0;
        wait(bvalid);
        repeat(2)begin @(negedge clk);if(!bvalid || bresp!=0)$fatal(1,"B stability");end
        bready=1;@(posedge clk);@(negedge clk);bready=0;
    end endtask
    task automatic axi_read(input logic[7:0]addr,output logic[31:0]data);begin
        @(negedge clk);araddr=addr;arvalid=1;rready=0;
        do @(posedge clk); while(!arready);
        @(negedge clk);arvalid=0;wait(rvalid);data=rdata;
        repeat(2)begin @(negedge clk);if(!rvalid || rdata!==data)$fatal(1,"R stability");end
        rready=1;@(posedge clk);@(negedge clk);rready=0;
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
    task automatic submit(input integer task_id,input bit w4,input integer dst);
        axi_write('h00,0);
        axi_write('h08,'h100); axi_write('h0c,w4?'h500:'h400);
        axi_write('h10,dst); axi_write('h1c,12);
        axi_write('h20,task_id==4?1:3); axi_write('h24,3); axi_write('h28,2);
        axi_write('h2c,task_id+100); axi_write('h30,task_id);
        axi_write('h00,w4?'h801:1);
    endtask
    initial begin : test
        logic [31:0] status, value;
        integer ack;
        for(int i=0;i<4096;i++)mem[i]=0;
        awaddr=0;awvalid=0;wdata=0;wstrb=0;wvalid=0;bready=0;
        araddr=0;arvalid=0;rready=0;
        put_qword('h100,64'h0000000000050301);
        put_qword('h108,64'h0000000000060402);
        put_qword('h400,64'h0000000000fefd07);
        put_qword('h408,64'h0000000000030504);
        // W4 vectors occupy four bytes per K step.
        put_qword('h500,64'h0000035400000ed7);
        repeat(20)@(negedge clk);rst_n=1;
        axi_read('h40,value);
        if(value!==32'h4c4c4d01)$fatal(1,"ABI identity");
        submit(1,0,'h800);
        // Completion is held while four additional commands fill the FIFO.
        submit(2,1,'h840);submit(3,0,'h880);
        submit(4,1,'h8c0);submit(5,0,'h900);
        submit(99,0,'ha00); // full: explicit rejection, never executed
        axi_read('h04,status);
        if(!status[4] || status[7:5]!=4)$fatal(1,"full rejection/occupancy %h",status);
        ack=0;
        for(int id=1;id<=5;id++)begin
            status=0;
            while(!status[3])axi_read('h04,status);
            axi_read('h44,value);if(value!==id+100)$fatal(1,"batch mismatch");
            axi_read('h48,value);if(value!==id)$fatal(1,"task mismatch");
            axi_read('h4c,value);if(value[2] || !value[1])$fatal(1,"backend status %h",value);
            // Check DDR-visible output before ACK; immutable completion owns it.
            expect_i32('h800+(id-1)*64,15);expect_i32('h804+(id-1)*64,7);
            expect_i32('h808+(id-1)*64,4);expect_i32('h80c+(id-1)*64,id==4?0:37);
            expect_i32('h810+(id-1)*64,id==4?0:11);expect_i32('h814+(id-1)*64,id==4?0:6);
            expect_i32('h818+(id-1)*64,id==4?0:59);expect_i32('h81c+(id-1)*64,id==4?0:15);
            expect_i32('h820+(id-1)*64,id==4?0:8);
            repeat(10)@(negedge clk);
            axi_read('h48,value);if(value!==id)$fatal(1,"completion overwritten");
            ack=1-ack;axi_write('h80,ack);
        end
        repeat(50)@(negedge clk);
        axi_read('h04,status);
        if(status & 32'h1fa)$fatal(1,"queue not drained %h",status);
        expect_i32('ha00,0);
        // Unsupported opcode returns a tagged error without touching backend.
        axi_write('h00,0);axi_write('h30,123);axi_write('h00,'h40001);
        status=0;while(!status[3])axi_read('h04,status);
        axi_read('h48,value);if(value!=123)$fatal(1,"reject tag");
        axi_read('h4c,value);if(value!=32'h80000001)$fatal(1,"opcode reject");
        ack=1-ack;axi_write('h80,ack);
        // Invalid K reaches the existing scheduler and latches hardware fault.
        axi_write('h00,0);axi_write('h28,0);axi_write('h30,124);axi_write('h00,1);
        status=0;while(!status[3])axi_read('h04,status);
        if(!status[8])$fatal(1,"hardware fault not latched");
        axi_read('h4c,value);if(!value[2])$fatal(1,"hardware fault status");
        ack=1-ack;axi_write('h80,ack);
        submit(125,0,'ha40);
        repeat(200)@(negedge clk);
        axi_read('h04,status);
        if(!status[8] || status[3] || status[7:5]!=1)$fatal(1,"dispatch continued after fault");
        expect_i32('ha40,0);
        // Reset explicitly cancels queued and in-flight state.
        @(negedge clk);rst_n=0;repeat(3)@(negedge clk);rst_n=1;
        axi_read('h04,status);if(status & 32'h1fa)$fatal(1,"reset state");
        if(errors)$fatal(1,"golden failures %0d",errors);
        $display("PASS: queued AXI W8/W4 GEMM, DDR goldens, tags, full rejection, completion ACK");
        $finish;
    end
    initial begin #2000000; $fatal(1,"queued AXI timeout"); end
endmodule
