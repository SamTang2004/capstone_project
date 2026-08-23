module tb_layernorm_int8;
    logic clk=0,rst_n=0,start=0,input_valid=0,input_ready,param_valid=0,param_ready;
    logic output_valid,output_ready=0,output_last,busy,done,error;
    logic [2:0] vector_length;
    logic [31:0] epsilon_term=0;
    logic signed [7:0] input_data,output_data;
    logic signed [15:0] gamma_q8=256,beta_q8=0;
    logic signed [7:0] inputs[0:3];
    logic signed [7:0] expected[0:3];
    integer errors=0;
    always #5 clk=~clk;
    layernorm_int8 #(.MAX_LENGTH(4)) dut(.*);
    initial begin
        inputs[0]=-3;inputs[1]=-1;inputs[2]=1;inputs[3]=3;
        expected[0]=-2;expected[1]=-1;expected[2]=1;expected[3]=2;
        repeat(3)@(posedge clk);rst_n=1;
        @(negedge clk);vector_length=4;start=1;@(negedge clk);start=0;
        for(int i=0;i<4;i++)begin input_data=inputs[i];input_valid=1;@(posedge clk);@(negedge clk);end
        input_valid=0;param_valid=1;output_ready=1;
        for(int i=0;i<4;i++)begin wait(output_valid);#1;if(output_data!==expected[i]||output_last!==(i==3))begin
            $error("layernorm index=%0d expected=%0d got=%0d last=%0b sum=%0d ss=%0d denom=%0d sample=%0d",i,expected[i],output_data,output_last,dut.sum,dut.sum_squares,dut.denominator,dut.samples[dut.index]);errors++;end
            @(posedge clk);@(negedge clk);
        end
        param_valid=0;output_ready=0;wait(done);
        if(error||errors)$fatal(1,"LayerNorm failed errors=%0d",errors);
        $display("LayerNorm INT8 golden trace passed");$finish;
    end
endmodule
