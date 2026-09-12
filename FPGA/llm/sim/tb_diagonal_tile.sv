module tb_diagonal_tile;
    logic clk=0,rst_n=0; always #5 clk=~clk;
    logic cmd_valid=0,cmd_ready,cmd_w4=0;
    logic [15:0] cmd_batch=9,cmd_k=0;
    logic [31:0] cmd_tile_id=0;
    logic [3:0] cmd_lanes=0;
    logic [255:0] cmd_lane_tags=0;
    logic operand_valid=0,operand_ready;
    logic [63:0] operand_a=0,operand_b=0;
    logic completion_valid,completion_ready=0;
    logic [15:0] completion_batch;
    logic [31:0] completion_tile_id,execution_cycles,input_wait_cycles;
    logic [1:0] completion_status;
    logic [7:0] result_mask;
    logic [255:0] result_tags,result_values;
    logic [2047:0] debug_accumulators;
    integer ticks=0,trace,raw_cycles=0,compact_cycles=0;
    always @(posedge clk) ticks<=ticks+1;
    llm_diagonal_tile dut(.*);

    task automatic run_case(input integer id, lanes, steps, bubbles);
        integer a[0:15][0:7],b[0:15][0:7],gold[0:7][0:7];
        integer started,finished;
        logic [255:0] saved_tags,saved_values;
        logic [7:0] saved_mask;
        for(int k=0;k<16;k++)for(int r=0;r<8;r++)begin a[k][r]=0;b[k][r]=0;end
        if(id==0)begin
            a[0][0]=1;a[1][0]=2;a[2][0]=1;
            b[0][0]=3;b[1][0]=4;
        end else if(id==1)begin
            a[0][0]=5;a[2][0]=6;b[0][0]=7;b[2][0]=8;
        end else if(id==2)begin
            a[0][0]=1;a[1][0]=2;b[0][0]=3;b[1][0]=4;
        end else if(id==3)begin
            a[0][0]=5;a[1][0]=6;b[0][0]=7;b[1][0]=8;
        end else if(id==7)begin
            for(int r=0;r<8;r++)begin
                a[0][r]=r-4;b[0][r]=r+1;
                a[1][r]=2*r-7;b[1][r]=3-r;
            end
        end else begin
            a[0][0]=1;a[1][0]=2;b[0][0]=3;b[1][0]=4;
            a[0][1]=5;a[1][1]=6;b[0][1]=7;b[1][1]=8;
        end
        for(int r=0;r<8;r++)for(int c=0;c<8;c++)begin
            gold[r][c]=0;
            for(int k=0;k<steps;k++)gold[r][c]+=a[k][r]*b[k][c];
        end
        @(negedge clk);if(!cmd_ready)$fatal(1,"command not ready");
        cmd_tile_id=id;cmd_batch=9;cmd_lanes=lanes;cmd_k=steps;cmd_lane_tags=0;
        for(int r=0;r<lanes;r++)cmd_lane_tags[r*32+:32]=100+r;
        if(id==1 || id==3)cmd_lane_tags[0+:32]=101;
        saved_tags=cmd_lane_tags;cmd_valid=1;
        @(posedge clk);#1;started=ticks;
        @(negedge clk);cmd_valid=0;
        // Descriptor writes after acceptance must not alter the active tags.
        cmd_lane_tags='1;cmd_tile_id='1;cmd_batch='1;cmd_lanes=8;
        wait(operand_ready);
        repeat(bubbles)@(negedge clk);
        for(int k=0;k<steps;k++)begin
            @(negedge clk);operand_a=0;operand_b=0;operand_valid=1;
            for(int r=0;r<8;r++)begin operand_a[r*8+:8]=a[k][r];operand_b[r*8+:8]=b[k][r];end
            do @(posedge clk);while(!operand_ready);
            #1;
            $fwrite(trace,"{\"event\":\"input\",\"id\":%0d,\"tick\":%0d,\"a\":\"%016h\",\"b\":\"%016h\"}\n",id,ticks,operand_a,operand_b);
        end
        @(negedge clk);operand_valid=0;
        wait(completion_valid);#1;finished=ticks;
        if(completion_status || completion_tile_id!=id || completion_batch!=9)$fatal(1,"completion identity");
        if(execution_cycles!=steps+4+bubbles || finished-started!=execution_cycles || input_wait_cycles!=bubbles)
            $fatal(1,"cycle mismatch id=%0d exec=%0d edges=%0d waits=%0d",id,execution_cycles,finished-started,input_wait_cycles);
        if(result_mask!==((1<<lanes)-1) || result_tags!==saved_tags)$fatal(1,"lane tags/mask");
        for(int r=0;r<8;r++)for(int c=0;c<8;c++)
            if($signed(debug_accumulators[(r*8+c)*32+:32])!==gold[r][c])$fatal(1,"matrix golden");
        for(int r=0;r<8;r++)
            if($signed(result_values[r*32+:32])!==(r<lanes?gold[r][r]:0))$fatal(1,"tag scatter golden");
        if(id>=4 && id<=6 && (gold[0][0]!=11 || gold[1][1]!=83 || gold[0][1]!=23 || gold[1][0]!=39))$fatal(1,"example golden");
        $fwrite(trace,"{\"event\":\"completion\",\"id\":%0d,\"k\":%0d,\"lanes\":%0d,\"start_tick\":%0d,\"finish_tick\":%0d,\"execution_cycles\":%0d,\"wait_cycles\":%0d,\"results\":[",id,steps,lanes,started,finished,execution_cycles,input_wait_cycles);
        for(int r=0;r<lanes;r++)begin
            if(r!=0)$fwrite(trace,",");
            $fwrite(trace,"{\"tag\":%0d,\"value\":%0d}",result_tags[r*32+:32],$signed(result_values[r*32+:32]));
        end
        $fwrite(trace,"]}\n");
        $display("CASE id=%0d K=%0d lanes=%0d execution=%0d waits=%0d",id,steps,lanes,execution_cycles,input_wait_cycles);
        if(id<2)raw_cycles+=execution_cycles;
        else if(id<4)compact_cycles+=execution_cycles;
        saved_values=result_values;saved_mask=result_mask;
        repeat(4)begin
            @(negedge clk);
            if(!completion_valid || result_values!==saved_values || result_tags!==saved_tags || result_mask!==saved_mask || cmd_ready)
                $fatal(1,"completion backpressure");
        end
        completion_ready=1;@(negedge clk);completion_ready=0;
    endtask

    initial begin
        trace=$fopen("diagonal_trace.jsonl","w");if(!trace)$fatal(1,"trace open");
        repeat(20)@(negedge clk);rst_n=1;
        run_case(0,1,4,0);run_case(1,1,4,0);
        run_case(2,1,2,0);run_case(3,1,2,0);
        run_case(4,2,2,0);run_case(5,2,16,0);
        run_case(6,2,2,3);run_case(7,8,2,0);
        if(raw_cycles!=16 || compact_cycles!=12)$fatal(1,"baseline totals");
        $fclose(trace);
        $display("PASS: diagonal RTL raw=16 compact=12 fused=6 padded=20; tags, signed goldens, stalls, backpressure");
        $finish;
    end
    initial begin #200000;$fatal(1,"timeout");end
endmodule
