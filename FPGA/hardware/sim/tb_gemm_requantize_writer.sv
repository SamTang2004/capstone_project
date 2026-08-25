module tb_gemm_requantize_writer;
    logic clk=0,rst_n=0,pipeline_enable=1,start=0;
    logic [2047:0] accumulators='0;
    logic [255:0] bias_values='0;
    logic bias_enable=1;
    logic [3:0] valid_rows=8,valid_cols=2;
    logic [1:0] output_mode=0;
    logic [31:0] output_row_stride=2;
    logic [31:0] output_base_addr=32'h100;
    logic signed [31:0] multiplier=2,output_zero_point=0;
    logic [5:0] right_shift=1;
    logic busy,done,capture_done,error,write_req_valid,write_req_ready=1,write_req_last;
    logic [31:0] write_req_addr,write_req_data;
    logic [3:0] write_req_strb;
    logic [7:0] mem[0:511];
    integer errors=0, launch_cycles, capture_cycle;
    always #5 clk=~clk;

    gemm_requantize_writer dut(.*);

    always_ff @(posedge clk) if(write_req_valid&&write_req_ready)
        for(int i=0;i<4;i++) if(write_req_strb[i])
            mem[write_req_addr+i]<=write_req_data[i*8+:8];

    task automatic expect_byte(input int address,input int expected);
        if($signed(mem[address])!==expected)begin
            $error("byte[%0h] expected=%0d got=%0d",address,expected,$signed(mem[address]));errors++;
        end
    endtask

    task automatic launch;
        begin
            launch_cycles=0;capture_cycle=-1;
            @(negedge clk);start=1;@(negedge clk);start=0;
            while(!done)begin
                @(posedge clk);launch_cycles++;
                if(capture_done)capture_cycle=launch_cycles;
            end
            if(capture_cycle<0||capture_cycle>38)begin
                $error("two-lane capture exceeded bound cycles=%0d",capture_cycle);errors++;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        for(int i=0;i<512;i++)mem[i]=8'hcc;
        for(int row=0;row<8;row++)begin
            accumulators[(row*8+0)*32+:32]=2*row+1;
            accumulators[(row*8+1)*32+:32]=2*row+2;
        end
        bias_values[0+:32]=10;bias_values[32+:32]=-2;
        repeat(3)@(posedge clk);rst_n<=1;launch();
        for(int row=0;row<8;row++)begin
            expect_byte('h100+row,2*row+11);
            expect_byte('h108+row,2*row);
        end

        // Dense row-major mode must handle odd strides and aligned partial writes.
        bias_enable=0;valid_rows=3;valid_cols=7;output_mode=1;
        output_base_addr='h180;output_row_stride=7;
        for(int row=0;row<3;row++)for(int col=0;col<7;col++)
            accumulators[(row*8+col)*32+:32]=row*16+col;
        launch();
        for(int row=0;row<3;row++)for(int col=0;col<7;col++)
            expect_byte('h180+row*7+col,row*16+col);

        // Dynamic-B mode stores one padded eight-column vector per reduction row.
        valid_rows=3;valid_cols=5;output_mode=2;output_base_addr='h040;output_row_stride=8;
        launch();
        for(int row=0;row<3;row++)for(int col=0;col<5;col++)
            expect_byte('h040+row*8+col,row*16+col);
        if(error||errors)$fatal(1,"Fused GEMM requant writer failed errors=%0d",errors);
        $display("Fused GEMM tiled, row-major, and dynamic-B golden trace passed");$finish;
    end
endmodule
