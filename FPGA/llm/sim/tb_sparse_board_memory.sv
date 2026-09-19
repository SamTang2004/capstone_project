`timescale 1ns/1ps
module tb_sparse_board_memory;
    logic clk=0;always #2.5 clk=~clk;
    logic rst_n=0;logic [31:0] ctrl=0;logic [511:0] staging=0;
    wire [31:0] status;wire [511:0] perf;
    integer cycles=0,checked=0;
    bit stall_enable=0;
    always @(negedge clk)begin
        if(stall_enable)begin
            if(cycles%41<23)force dut.engine.core.in_ready=1'b0;
            else force dut.engine.core.in_ready=dut.engine.core.rst_n && dut.engine.core.active && !dut.engine.core.last_seen;
        end else release dut.engine.core.in_ready;
    end
    always @(posedge clk)if(dut.engine_reset)begin
        if(dut.inflight>12 || dut.queue_count>12)$fatal(1,"response credit overflow");
    end
    integer x[0:4095];
    llm_sparse_board_memory dut(.*);
    function automatic integer weight_value(input integer row,col);
        weight_value=((row*13+col*7)%255)-127;
    endfunction
    task automatic tick;
        @(posedge clk);#1;@(negedge clk);cycles++;
        if(cycles>3000000)$fatal(1,"board harness timeout");
    endtask
    task automatic upload(input bit weights,input integer addr,input logic [31:0] data);
        staging[4*32+:32]=addr;staging[5*32+:32]=data;
        ctrl[weights?2:1]=~ctrl[weights?2:1];tick();
    endtask
    task automatic run_case(input integer n,nrows,density,xshift,input bit skip,reuse);
        logic [31:0] word_data;
        integer v,waited;
        longint signed golden;
        for(integer i=0;i<4096;i++)x[i]=(i<n && (i*137)%1000<density) ? ((i*17)%255)-128 : 0;
        for(integer i=0;i<(n+3)/4;i++)begin
            word_data=0;for(integer b=0;b<4;b++)word_data[b*8+:8]=8'(x[i*4+b]);
            upload(0,i,word_data);
        end
        staging[2*32+:32]=n;staging[3*32+:32]=nrows|(xshift<<8)|(skip<<16)|(reuse<<17);
        ctrl[0]=~ctrl[0];tick();waited=0;
        while(!status[1])begin tick();waited++;if(waited>150000)$fatal(1,"command timeout inflight=%0d queued=%0d ready=%0d last=%0d",dut.inflight,dut.queue_count,dut.rsp_ready,dut.engine.core.last_seen);end
        if(status[2] || status[0])$fatal(1,"board status error");
        for(integer row=0;row<nrows;row++)begin
            golden=0;
            for(integer i=0;i<n;i++)begin
                v=x[i];if((v<0?-v:v)>=(1<<xshift))golden+=v*weight_value(row,i);
            end
            staging[6*32+:32]=row;repeat(2)tick();
            if(perf[160+:32]!==golden[31:0])$fatal(1,"board row mismatch row=%0d got=%0d expected=%0d",row,$signed(perf[160+:32]),golden);
        end
        checked++;
        $display("BOARD_BENCH K=%0d rows=%0d density=%0d shift=%0d skip=%0d reuse=%0d cycles=%0d prep=%0d",n,nrows,density,xshift,skip,reuse,perf[32+:32],perf[64+:32]);
    endtask
    initial begin
        logic [31:0] word_data;
        repeat(30)tick();rst_n=1;tick();
        if(perf[0+:32]!=32'h53505202)$fatal(1,"identity");
        for(integer r=0;r<64;r++)for(integer c=0;c<4096;c+=4)begin
            for(integer b=0;b<4;b++)word_data[b*8+:8]=8'(weight_value(r,c+b));
            upload(1,r*1024+c/4,word_data);
        end
        for(integer mode=0;mode<4;mode++)begin
            run_case(65,9,500,0,mode[0],mode[1]);
            run_case(4096,64,1000,0,mode[0],mode[1]);
            run_case(4096,64,1,0,mode[0],mode[1]);
        end
        run_case(4096,64,100,0,1,1);run_case(4096,64,10,0,1,1);
        run_case(0,9,0,0,1,1);run_case(3,1,1000,0,1,1);
        run_case(129,17,1000,2,1,1);run_case(129,17,1000,8,1,1);
        run_case(129,17,1000,15,1,1);
        // Cancel outstanding memory work and immediately reuse the wrapper.
        staging[2*32+:32]=4096;staging[3*32+:32]=64;ctrl[0]=~ctrl[0];repeat(20)tick();
        ctrl[3]=~ctrl[3];repeat(2)tick();if(status[0])$fatal(1,"cancel failed");
        run_case(65,9,1000,0,1,1);
        // Invalidate each newly added weight-pipeline stage while it is live.
        for(integer depth=0;depth<4;depth++)begin
            staging[2*32+:32]=4096;staging[3*32+:32]=64;ctrl[0]=~ctrl[0];tick();
            while(!dut.address_valid)tick();
            repeat(depth)tick();ctrl[3]=~ctrl[3];repeat(2)tick();
            if(status[0])$fatal(1,"pipeline cancel failed depth=%0d",depth);
            run_case(65,9,1000,0,1,1);
        end
        // Stall the actual core handshake, fill the reserved response queue,
        // and verify that the complete dot products still match after release.
        stall_enable=1;run_case(4096,64,1000,0,1,1);stall_enable=0;
        staging[2*32+:32]=4097;ctrl[0]=~ctrl[0];tick();
        if(!status[2] || status[0])$fatal(1,"bad shape accepted");
        $display("PASS: sparse board memory commands=%0d",checked);$finish;
    end
endmodule
