module tb_llm_command_queue;
    logic clk=0, rst_n=0;
    always #5 clk=~clk;
    logic push_valid=0,push_ready,pop_valid,pop_ready=0;
    logic [127:0] push_data=0,pop_data;
    logic [1:0] occupancy;
    // Non-power-of-two depth deliberately exercises explicit pointer wrap.
    llm_command_queue #(.DEPTH(3)) dut(.*);
    function automatic [127:0] descriptor(input integer sequence_id);
        descriptor={32'habc01234,32'(sequence_id),32'hfedc9876,32'(sequence_id*7)};
    endfunction
    initial begin #10000; $fatal(1,"queue timeout"); end
    initial begin
        repeat(3) @(negedge clk); rst_n=1; #1;
        for(integer i=0;i<3;i++) begin
            push_valid=1; push_data=descriptor(i);
            if(!push_ready) $fatal(1,"early full");
            @(negedge clk);
        end
        push_data=descriptor(99); // held producer cannot overwrite full queue
        repeat(3) begin
            if(push_ready || !pop_valid || occupancy!=3 || pop_data!==descriptor(0))
                $fatal(1,"full/backpressure violation");
            @(negedge clk);
        end
        // Full simultaneous push/pop: preserve occupancy and all descriptor bits.
        pop_ready=1;
        for(integer i=0;i<9;i++) begin
            push_data=descriptor(i+3); #1;
            if(!push_ready || pop_data!==descriptor(i) || occupancy!=3)
                $fatal(1,"simultaneous FIFO routing mismatch %0d",i);
            @(negedge clk);
        end
        push_valid=0;
        for(integer i=9;i<12;i++) begin
            if(pop_data!==descriptor(i)) $fatal(1,"drain ordering");
            @(negedge clk);
        end
        if(pop_valid || occupancy!=0) $fatal(1,"empty violation");
        pop_ready=0; push_valid=1; push_data=descriptor(42);
        @(negedge clk); push_valid=0; rst_n=0;
        @(negedge clk); rst_n=1; #1;
        if(pop_valid || occupancy!=0) $fatal(1,"reset did not cancel queued work");
        $display("PASS: LLM command queue ordering, tags, full replacement, wrap, reset");
        $finish;
    end
endmodule
