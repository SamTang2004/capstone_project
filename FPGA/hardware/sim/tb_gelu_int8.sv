module tb_gelu_int8;
    logic input_valid=1,output_ready=1,input_ready,output_valid;
    logic [63:0] input_vector,output_vector;
    gelu_int8 dut(.*);
    initial begin
        input_vector={8'sd127,8'sd32,8'sd24,8'sd8,8'sd0,-8'sd8,-8'sd24,-8'sd32};#1;
        if(output_vector!=={8'sd127,8'sd32,8'sd24,8'sd5,8'sd0,-8'sd3,8'sd0,8'sd0})
            $fatal(1,"GELU mismatch got=%h",output_vector);
        $display("GELU INT8 golden trace passed");$finish;
    end
endmodule
