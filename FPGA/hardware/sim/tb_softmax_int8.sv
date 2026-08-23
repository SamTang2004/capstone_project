module tb_softmax_int8;
    logic clk=0,rst_n=0,start=0,input_valid=0,input_ready,output_valid,output_ready=0;
    logic output_last,busy,done,error;
    logic [2:0] vector_length;
    logic signed [7:0] input_data;
    logic [7:0] output_probability_q7;
    integer errors=0,sum=0;
    always #5 clk=~clk;
    softmax_int8 #(.MAX_LENGTH(4)) dut(.*);
    initial begin
        repeat(3)@(posedge clk);rst_n=1;
        @(negedge clk);vector_length=4;start=1;@(negedge clk);start=0;
        for(int i=0;i<4;i++)begin input_data=0;input_valid=1;@(posedge clk);@(negedge clk);end
        input_valid=0;output_ready=1;
        for(int i=0;i<4;i++)begin wait(output_valid);#1;
            if(output_probability_q7!==8'd32||output_last!==(i==3))begin $error("softmax index=%0d p=%0d last=%0b",i,output_probability_q7,output_last);errors++;end
            sum+=output_probability_q7;@(posedge clk);@(negedge clk);
        end
        output_ready=0;wait(done);
        if(sum!=128||error||errors)$fatal(1,"Softmax failed errors=%0d sum=%0d",errors,sum);
        $display("Softmax INT8 golden trace passed");$finish;
    end
endmodule
