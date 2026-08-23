module tb_rowwise_norm_softmax_engine;
    logic clk=0,rst_n=0,start=0,softmax_mode=0,input_valid=0,input_ready;
    logic param_valid=0,param_ready,output_valid,output_ready=0,output_last;
    logic busy,done,error;
    logic [2:0] vector_length;
    logic [31:0] epsilon_term=0;
    logic signed [7:0] input_data,output_data;
    logic signed [15:0] gamma_q8=256,beta_q8=0;
    logic signed [7:0] values[0:3];
    logic signed [7:0] expected_ln[0:3];
    integer errors=0,cycle_count=0,start_cycle,elapsed,sum;

    always #5 clk=~clk;
    always_ff @(posedge clk) cycle_count<=cycle_count+1;
    rowwise_norm_softmax_engine #(.MAX_LENGTH(4)) dut(.*);

    task automatic launch(input logic mode); begin
        softmax_mode=mode;vector_length=4;
        @(negedge clk);start=1;start_cycle=cycle_count;@(negedge clk);start=0;
        for(int i=0;i<4;i++)begin
            wait(input_ready);input_data=values[i];input_valid=1;
            @(posedge clk);@(negedge clk);
        end
        input_valid=0;param_valid=!mode;output_ready=1;sum=0;
        for(int i=0;i<4;i++)begin
            wait(output_valid);#1;
            if(mode)begin
                if(output_data!==8'sd32)begin
                    $error("softmax index=%0d expected=32 got=%0d",i,output_data);errors++;
                end
                sum+=output_data;
            end else if(output_data!==expected_ln[i])begin
                $error("layernorm index=%0d expected=%0d got=%0d",i,expected_ln[i],output_data);errors++;
            end
            if(output_last!==(i==3))begin $error("last mismatch index=%0d",i);errors++;end
            @(posedge clk);@(negedge clk);
        end
        param_valid=0;output_ready=0;wait(done);elapsed=cycle_count-start_cycle;
        if(mode&&sum!=128)begin $error("softmax sum expected=128 got=%0d",sum);errors++;end
        if(elapsed>300)begin $error("shared engine unexpectedly slow mode=%0d cycles=%0d",mode,elapsed);errors++;end
        @(posedge clk);
    end endtask

    initial begin
        values[0]=-3;values[1]=-1;values[2]=1;values[3]=3;
        expected_ln[0]=-2;expected_ln[1]=-1;expected_ln[2]=1;expected_ln[3]=2;
        repeat(3)@(posedge clk);rst_n=1;
        launch(0);
        for(int i=0;i<4;i++)values[i]=0;
        launch(1);
        if(error||errors)$fatal(1,"shared row engine failed errors=%0d",errors);
        $display("Shared LayerNorm/Softmax reciprocal-engine golden trace passed");$finish;
    end
    initial begin #100000;$fatal(1,"timeout state=%0d",dut.state);end
endmodule
