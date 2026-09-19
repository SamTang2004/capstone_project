`timescale 1ns/1ps
module tb_bitmap_finder64;
    logic clk=0; always #5 clk=~clk;
    logic rst_n=0,in_valid=0,out_ready=0;
    wire in_ready,out_valid,out_frame_last;
    logic [63:0] in_mask=0;
    logic [511:0] in_values=0;
    logic [3:0] in_shift=0;
    logic [12:0] in_meta=0;
    wire [12:0] out_meta;
    wire [3:0] out_count;
    wire [63:0] out_values;
    wire [47:0] out_source_indices;
    llm_bitmap_finder64 dut(.*);
    integer indices[0:63],values[0:63];
    integer n,pos,cycle=0,frames=0,survivors=0,stalls=0;
    bit done,accepted,held=0;
    logic [129:0] previous;
    wire [129:0] packet={out_count,out_values,out_source_indices,out_meta,out_frame_last};
    task automatic tick;
        @(posedge clk);
        accepted=in_valid && in_ready;
        if(rst_n && held && (!out_valid || packet!==previous)) $fatal(1,"stalled bitmap output lost/changed");
        if(rst_n && out_valid) begin
            if(out_count!=(n==0?0:1) || out_meta!==in_meta || out_frame_last!=(n==0 || pos==n-1))
                $fatal(1,"bitmap count/meta/end");
            if(n!=0 && (out_source_indices[5:0]!=indices[pos] || $signed(out_values[7:0])!=values[pos]))
                $fatal(1,"bitmap survivor mismatch");
            if(out_values[63:8]!=0 || out_source_indices[47:6]!=0) $fatal(1,"dirty unused slots");
            if(n==0 && (out_values!=0 || out_source_indices!=0)) $fatal(1,"dirty empty marker");
            if(out_ready) begin pos+=out_count; if(out_frame_last) done=1; end
            else stalls++;
        end
        held=rst_n && out_valid && !out_ready; previous=packet;
        cycle++; #1; @(negedge clk);
    endtask
    task automatic run_frame;
        integer v;
        n=0;pos=0;done=0;
        for(integer i=0;i<64;i++) begin
            v=$signed(in_values[i*8+:8]);
            if(in_mask[i] && (v<0?-v:v)>=(1<<in_shift)) begin indices[n]=i;values[n]=v;n++; end
        end
        in_valid=1;out_ready=0;do tick();while(!accepted);in_valid=0;
        while(!done) begin out_ready=cycle%3!=0;tick();end
        frames++;survivors+=n;in_meta++;
    endtask
    initial begin
        repeat(3)tick();rst_n=1;
        for(integer s=0;s<16;s++) begin
            in_shift=4'(s);in_mask='1;
            for(integer sample=0;sample<256;sample++) begin
                for(integer i=0;i<64;i++) in_values[i*8+:8]=8'(sample+i*17);
                run_frame();
            end
        end
        in_shift=0;in_values={64{8'h80}};
        for(integer i=0;i<64;i++) begin in_mask=64'd1<<i;run_frame();end
        for(integer count=0;count<=64;count++) begin
            in_mask=0;
            for(integer i=0;i<count;i++)in_mask[(i*17)%64]=1;
            run_frame();
        end
        // Cancel during each preparation stage and with a stalled output.
        for(integer depth=0;depth<7;depth++) begin
            in_mask='1;in_values={64{8'h80}};in_shift=0;
            n=64;pos=0;done=0;
            for(integer i=0;i<64;i++)begin indices[i]=i;values[i]=-128;end
            out_ready=0;in_valid=1;do tick();while(!accepted);in_valid=0;
            repeat(depth)tick();rst_n=0;tick();rst_n=1;
            repeat(7)begin tick();if(out_valid)$fatal(1,"finder stale result after reset");end
            in_mask=64'h8000000000000001;run_frame();
        end
        $display("PASS: bitmap finder frames=%0d survivors=%0d stalls=%0d cycles=%0d reset_depths=7",frames,survivors,stalls,cycle);
        $finish;
    end
endmodule
