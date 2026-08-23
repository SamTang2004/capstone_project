module tb_bias_add_int32;
    logic input_valid=1,output_ready=1,input_ready,output_valid;
    logic [255:0] accumulators,bias,result;
    bias_add_int32 dut(.*);
    initial begin
        for(int lane=0;lane<8;lane++)begin
            accumulators[lane*32+:32]=$signed(lane*100-250);
            bias[lane*32+:32]=$signed(20-lane);
        end
        #1;
        for(int lane=0;lane<8;lane++)if($signed(result[lane*32+:32])!==lane*99-230)
            $fatal(1,"bias add mismatch lane=%0d got=%0d",lane,$signed(result[lane*32+:32]));
        $display("Bias-add INT32 golden trace passed");$finish;
    end
endmodule
