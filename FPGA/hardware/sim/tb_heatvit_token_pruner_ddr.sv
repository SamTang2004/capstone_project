`timescale 1ns/1ps
module tb_heatvit_token_pruner_ddr;
    logic clk=0,rst_n=0,pipeline_enable=1,start=0,weight_int4=0;
    logic [31:0] input_base_addr=32'h100,weight_base_addr=32'h200,output_base_addr=32'h300;
    logic [15:0] input_tokens=5,output_tokens=3,features=8;
    logic [31:0] package_reciprocal_q16=21845;
    logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
    logic read_req_valid,read_req_ready=1,read_req_wide,read_req_qword,read_req_allow_ahead;
    logic [31:0] read_req_addr; logic [6:0] read_req_len;
    logic read_data_valid=0,read_data_ready,read_data_last=0;
    logic [7:0] read_data;
    logic read_qword_valid=0,read_qword_ready,read_qword_last=0;
    logic [63:0] read_qword_data;logic[7:0]read_qword_keep;
    logic write_req_valid,write_req_ready=1,write_req_last,memory_busy=0,memory_error=0;
    logic[31:0]write_req_addr,write_req_data;logic[3:0]write_req_strb;
    logic [7:0] mem [0:1023];
    logic pending_byte,pending_qword;logic[31:0]pending_addr;
    integer i,j;
    always #5 clk=~clk;

    heatvit_token_pruner_ddr #(.MAX_TOKENS(8),.MAX_FEATURES(8)) dut(.*);

    always_ff @(posedge clk) begin
        read_data_valid<=0;read_qword_valid<=0;read_data_last<=0;read_qword_last<=0;
        if(read_req_valid&&read_req_ready)begin pending_addr<=read_req_addr;pending_byte<=!read_req_qword;pending_qword<=read_req_qword;end
        else begin pending_byte<=0;pending_qword<=0;end
        if(pending_byte)begin read_data<=mem[pending_addr];read_data_valid<=1;read_data_last<=1;end
        if(pending_qword)begin
            for(j=0;j<8;j=j+1)read_qword_data[j*8+:8]<=mem[pending_addr+j];
            read_qword_keep<=8'hff;read_qword_valid<=1;read_qword_last<=1;
        end
        if(write_req_valid&&write_req_ready)for(j=0;j<4;j=j+1)if(write_req_strb[j])mem[write_req_addr+j]<=write_req_data[j*8+:8];
    end

    task put_row(input integer row,input integer a,input integer b);
        begin
            mem['h100+row*8]=a[7:0];mem['h101+row*8]=b[7:0];
            for(i=2;i<8;i=i+1)mem['h100+row*8+i]=(row*8+i);
        end
    endtask
    initial begin
        for(i=0;i<1024;i=i+1)mem[i]=0;
        put_row(0,9,9);put_row(1,1,10);put_row(2,7,20);put_row(3,3,30);put_row(4,-2,40);
        mem['h200]=1; // W8 lane zero for feature zero; all other weights are zero.
        repeat(3)@(posedge clk);rst_n<=1;@(posedge clk);start<=1;@(posedge clk);start<=0;
        wait(done);repeat(3)@(posedge clk);
        if(error)$fatal(1,"pruner error");
        if($signed(mem['h300])!=9||$signed(mem['h308])!=7||$signed(mem['h310])!=1)$fatal(1,"token order/score mismatch");
        if($signed(mem['h301])!=9||$signed(mem['h309])!=20||$signed(mem['h311])!=27)$fatal(1,"package mismatch");
        $display("HeatViT token-pruner golden trace passed: cls=%0d selected=%0d package=[%0d,%0d]",$signed(mem['h300]),$signed(mem['h308]),$signed(mem['h310]),$signed(mem['h311]));
        $finish;
    end
    initial begin repeat(20000)@(posedge clk);$fatal(1,"timeout");end
endmodule
