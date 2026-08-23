module tb_transformer_debug_counters;
    logic clk=0, rst_n=0, clear=0, debug_enable=0, debug_step=0, freeze_counters=0;
    logic accelerator_busy=0, memory_stall=0, input_stall=0, output_stall=0;
    logic op_start=0, op_done=0, error_event=0;
    logic [7:0] op_code=0, active_op, trace_event;
    logic [31:0] debug_data=0, completed_ops, trace_data;
    logic pipeline_enable, error_latched, trace_valid;
    logic [63:0] total_cycles,busy_cycles,memory_stall_cycles,input_stall_cycles;
    logic [63:0] output_stall_cycles,last_op_cycles,max_op_cycles;
    integer errors=0;
    always #5 clk=~clk;
    transformer_debug_counters dut(.*);
    initial begin
        repeat(3)@(posedge clk); rst_n=1;
        @(negedge clk); debug_enable=1; op_code=8'h12; op_start=1; accelerator_busy=1;
        @(negedge clk); op_start=0; memory_stall=1;
        repeat(3)@(posedge clk);
        @(negedge clk); memory_stall=0; op_done=1;
        @(negedge clk); op_done=0; accelerator_busy=0;
        if(pipeline_enable!==0)begin $error("debug mode did not pause pipeline");errors++;end
        debug_step=1; #1; if(pipeline_enable!==1)begin $error("debug step did not enable pipeline");errors++;end
        @(negedge clk); debug_step=0; error_event=1; debug_data=32'hdeadbeef;
        @(posedge clk); #1;
        if(!error_latched||trace_event!==8'hff||trace_data!==32'hdeadbeef)begin $error("error trace mismatch");errors++;end
        error_event=0;
        if(completed_ops!==1||active_op!==8'h12||last_op_cycles==0||busy_cycles==0||memory_stall_cycles!=3)begin
            $error("counter mismatch ops=%0d last=%0d busy=%0d mem=%0d",completed_ops,last_op_cycles,busy_cycles,memory_stall_cycles);errors++;
        end
        if(errors)$fatal(1,"debug counters failed errors=%0d",errors);
        $display("Transformer debug-counter golden trace passed");$finish;
    end
endmodule
