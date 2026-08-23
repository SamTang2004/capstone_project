module tb_residual_add_int8;
    logic input_valid=1,output_ready=1,input_ready,output_valid;
    logic [63:0] lhs,rhs,result;
    logic [3:0] lhs_right_shift=0,rhs_right_shift=0;
    residual_add_int8 dut(.*);
    initial begin
        lhs={8'sd100,-8'sd100,8'sd127,-8'sd128,8'sd10,-8'sd10,8'sd1,-8'sd1};
        rhs={8'sd50,-8'sd50,8'sd1,-8'sd1,-8'sd20,8'sd20,-8'sd2,8'sd2};
        #1;
        if(result!=={8'sd127,-8'sd128,8'sd127,-8'sd128,-8'sd10,8'sd10,-8'sd1,8'sd1})
            $fatal(1,"residual add mismatch got=%h",result);
        lhs_right_shift=1;rhs_right_shift=1;#1;
        if($signed(result[63:56])!==75)$fatal(1,"scaled residual mismatch");
        $display("Residual-add INT8 golden trace passed");$finish;
    end
endmodule
