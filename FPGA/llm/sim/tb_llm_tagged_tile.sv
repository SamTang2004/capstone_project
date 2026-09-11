module tb_llm_tagged_tile;
    logic clk=0, rst_n=0;
    always #5 clk=~clk;
    logic cmd_valid=0, cmd_ready, cmd_w4=0;
    logic [15:0] cmd_batch=0, cmd_k=0;
    logic [31:0] cmd_task=0;
    logic [3:0] cmd_rows=0, cmd_cols=0;
    logic operand_valid=0, operand_ready;
    logic [63:0] operand_a=0, operand_b=0;
    logic completion_valid, completion_ready=0;
    logic [15:0] completion_batch;
    logic [31:0] completion_task, execution_cycles, input_wait_cycles;
    logic [1:0] completion_status;
    logic [2047:0] completion_accumulators;
    logic [31:0] baseline;
    llm_tagged_tile dut(.*);
    initial begin #100000; $fatal(1,"timeout"); end

    task automatic run_case(input bit w4, input bit invalid, input integer bubbles, input integer rows);
        logic [2047:0] saved;
        logic [31:0] cycles_saved;
        integer expected;
        @(negedge clk);
        if (!cmd_ready) $fatal(1,"command not ready");
        cmd_batch=w4?16'hcafe:16'h1234; cmd_task=32'habcd0000+bubbles;
        cmd_w4=w4; cmd_k=invalid?0:2; cmd_rows=rows; cmd_cols=3; cmd_valid=1;
        @(negedge clk); cmd_valid=0;
        if (!invalid) begin
            wait(operand_ready);
            repeat(bubbles) @(negedge clk);
            // A columns [1,3,5], [2,4,6]; B rows [7,-3,-2], [4,5,3].
            @(negedge clk); operand_valid=1; operand_a=64'h0000000000050301;
            operand_b=w4?64'hed7:64'h0000000000fefd07;
            @(negedge clk); operand_a=64'h0000000000060402;
            operand_b=w4?64'h354:64'h0000000000030504;
            @(negedge clk); operand_valid=0;
        end
        wait(completion_valid); #1;
        if(completion_batch!==cmd_batch || completion_task!==cmd_task)
            $fatal(1,"tag routing mismatch");
        if(completion_status!==(invalid?2'd1:2'd0)) $fatal(1,"status mismatch");
        if(!invalid) begin
            for(integer r=0;r<8;r++) for(integer c=0;c<8;c++) begin
                expected=0;
                if(r<rows && c<3) case(c)
                    0: expected=(1+2*r)*7+(2+2*r)*4;
                    1: expected=(1+2*r)*(-3)+(2+2*r)*5;
                    2: expected=(1+2*r)*(-2)+(2+2*r)*3;
                endcase
                if($signed(completion_accumulators[(r*8+c)*32+:32])!==expected)
                    $fatal(1,"golden mismatch W4=%0d row=%0d col=%0d",w4,r,c);
            end
            if(bubbles==0) baseline=execution_cycles;
            else if(execution_cycles != baseline+bubbles)
                $fatal(1,"bubble timing mismatch");
            if(input_wait_cycles != bubbles) $fatal(1,"input wait accounting: got %0d expected %0d",input_wait_cycles,bubbles);
        end else if(execution_cycles!=0) $fatal(1,"invalid command ran core");
        saved=completion_accumulators; cycles_saved=execution_cycles;
        repeat(4) begin
            @(negedge clk);
            if(!completion_valid || cmd_ready || saved!==completion_accumulators ||
               cycles_saved!==execution_cycles || completion_task!==cmd_task)
                $fatal(1,"completion unstable under backpressure");
        end
        completion_ready=1;
        @(negedge clk); completion_ready=0;
    endtask
    initial begin
        // Allow the vendor glbl power-on GSR (100 ns) to deassert first.
        repeat(20) @(negedge clk); rst_n=1;
        run_case(0,0,0,3);
        run_case(1,0,3,3);
        run_case(1,1,0,3);
        run_case(1,0,0,3);
        run_case(0,0,0,1);
        run_case(1,0,2,1);
        $display("PASS: LLM tagged tile W8/W4 goldens, tags, bubbles, backpressure, invalid recovery");
        $finish;
    end
endmodule
