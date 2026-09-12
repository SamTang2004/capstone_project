module tb_fusion_goldens;
    logic clk=0,rst_n=0; always #5 clk=~clk;
    logic cmd_valid=0,cmd_ready,cmd_w4=1;
    logic [15:0] cmd_batch=0,cmd_k=16;
    logic [31:0] cmd_task=0;
    logic [3:0] cmd_rows=0,cmd_cols=0;
    logic operand_valid=0,operand_ready;
    logic [63:0] operand_a=0,operand_b=0;
    logic completion_valid,completion_ready=0;
    logic [15:0] completion_batch;
    logic [31:0] completion_task,execution_cycles,input_wait_cycles;
    logic [1:0] completion_status;
    logic [2047:0] completion_accumulators;
    integer ticks=0;
    always @(posedge clk) ticks<=ticks+1;
    llm_tagged_tile dut(.*);
    initial begin : run_test
        integer src,trace,count,id,batch,rows,cols,k,rc,expected,launch_tick;
        logic [63:0] aa[0:15],bb[0:15];
        logic [2047:0] saved;
        src=$fopen("vectors.txt","r"); trace=$fopen("rtl_trace.jsonl","w");
        if(!src || !trace)$fatal(1,"fixture files");
        rc=$fscanf(src,"%d",count);if(rc!=1)$fatal(1,"fixture count");
        repeat(20)@(negedge clk);rst_n=1;
        for(integer tile=0;tile<count;tile++)begin
            rc=$fscanf(src,"%d %d %d %d %d",id,batch,rows,cols,k);
            if(rc!=5 || k!=16)$fatal(1,"fixture header");
            for(integer step=0;step<16;step++)begin
                rc=$fscanf(src,"%h %h",aa[step],bb[step]);
                if(rc!=2)$fatal(1,"fixture operand");
            end
            @(negedge clk);if(!cmd_ready)$fatal(1,"not ready");
            cmd_task=id;cmd_batch=batch;cmd_rows=rows;cmd_cols=cols;cmd_valid=1;
            @(posedge clk);launch_tick=ticks;
            @(negedge clk);cmd_valid=0;
            wait(operand_ready);
            for(integer step=0;step<16;step++)begin
                @(negedge clk);operand_valid=1;operand_a=aa[step];operand_b=bb[step];
                do @(posedge clk);while(!operand_ready);
            end
            @(negedge clk);operand_valid=0;
            wait(completion_valid);#1;
            if(completion_status || completion_task!=id || completion_batch!=batch)$fatal(1,"tag/status");
            $fwrite(trace,"{\"tile_id\":%0d,\"batch\":%0d,\"launch_tick\":%0d,\"completion_tick\":%0d,\"execution_cycles\":%0d,\"input_wait_cycles\":%0d,\"output\":[",id,batch,launch_tick,ticks,execution_cycles,input_wait_cycles);
            for(integer i=0;i<64;i++)begin
                rc=$fscanf(src,"%d",expected);
                if(rc!=1 || $signed(completion_accumulators[i*32+:32])!==expected)
                    $fatal(1,"tile %0d output %0d got %0d wanted %0d",id,i,$signed(completion_accumulators[i*32+:32]),expected);
                if(i!=0)$fwrite(trace,",");
                $fwrite(trace,"%0d",$signed(completion_accumulators[i*32+:32]));
            end
            $fwrite(trace,"]}\n");
            saved=completion_accumulators;
            repeat(3)begin
                @(negedge clk);
                if(!completion_valid || saved!==completion_accumulators)$fatal(1,"completion backpressure");
            end
            completion_ready=1;@(negedge clk);completion_ready=0;
        end
        $fclose(src);$fclose(trace);
        $display("PASS: fusion golden RTL tiles=%0d",count);$finish;
    end
    initial begin #2000000;$fatal(1,"fusion timeout");end
endmodule
