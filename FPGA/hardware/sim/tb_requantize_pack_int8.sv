module tb_requantize_pack_int8;
    logic clk=0,rst_n=0,pipeline_enable=1,start=0; always #5 clk=~clk;
    logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
    logic read_req_valid,read_req_ready=1,read_req_wide,read_req_qword,read_req_allow_ahead;
    logic [31:0] read_req_addr;logic [6:0]read_req_len;
    logic read_word_valid=0,read_word_ready,read_word_last=0;logic[31:0]read_word_data;logic[3:0]read_word_keep;
    logic write_req_valid,write_req_ready=1,write_req_last;logic[31:0]write_req_addr,write_req_data;logic[3:0]write_req_strb;
    logic memory_busy=0,memory_error=0;logic[7:0]mem[0:1023];integer errors=0;

    requantize_pack_int8 dut(.clk,.rst_n,.pipeline_enable,.start,
        .input_base_addr(32'h100),.output_base_addr(32'h300),.input_row_stride(32'd12),
        .rows(16'd3),.columns(16'd3),.multiplier(32'sd1),.right_shift(6'd0),.output_zero_point(32'sd0),
        .busy,.done,.error,.op_start,.op_done,.memory_stall,.input_stall,.output_stall,
        .read_req_valid,.read_req_ready,.read_req_addr,.read_req_len,.read_req_wide,.read_req_qword,.read_req_allow_ahead,
        .read_word_valid,.read_word_ready,.read_word_data,.read_word_keep,.read_word_last,
        .write_req_valid,.write_req_ready,.write_req_addr,.write_req_data,.write_req_strb,.write_req_last,.memory_busy,.memory_error);

    always_ff @(posedge clk) begin
        read_word_valid<=0;read_word_last<=0;
        if(read_req_valid&&read_req_ready)begin
            read_word_data<={mem[read_req_addr+3],mem[read_req_addr+2],mem[read_req_addr+1],mem[read_req_addr]};
            read_word_keep<=4'hf;read_word_valid<=1;read_word_last<=1;
        end
        if(write_req_valid&&write_req_ready)
            for(int lane=0;lane<4;lane++)if(write_req_strb[lane])mem[write_req_addr+lane]<=write_req_data[lane*8+:8];
    end
    task automatic put_i32(input int addr,input integer value);begin
        for(int lane=0;lane<4;lane++)mem[addr+lane]=value>>(lane*8);
    end endtask
    initial begin
        for(int i=0;i<1024;i++)mem[i]=0;
        put_i32('h100,1);put_i32('h104,-2);put_i32('h108,200);
        put_i32('h10c,4);put_i32('h110,5);put_i32('h114,6);
        put_i32('h118,-200);put_i32('h11c,8);put_i32('h120,9);
        repeat(4)@(posedge clk);rst_n=1;@(negedge clk);start=1;@(negedge clk);start=0;
        wait(done);@(posedge clk);
        if(error)begin $error("unexpected error");errors++;end
        // [M tile][column][8 row lanes], with saturation and zero padding.
        if({mem['h302],mem['h301],mem['h300]}!==24'h800401)begin $error("column0 packing failed");errors++;end
        if({mem['h30a],mem['h309],mem['h308]}!==24'h0805fe)begin $error("column1 packing failed");errors++;end
        if({mem['h312],mem['h311],mem['h310]}!==24'h09067f)begin $error("column2 packing failed");errors++;end
        for(int a='h303;a<'h308;a++)if(mem[a]!==0)errors++;
        for(int a='h30b;a<'h310;a++)if(mem[a]!==0)errors++;
        for(int a='h313;a<'h318;a++)if(mem[a]!==0)errors++;
        if(errors)$fatal(1,"Requantize-pack failed errors=%0d",errors);
        $display("INT32 to tile-major INT8 requantize-pack golden trace passed");$finish;
    end
    initial begin#100000;$fatal(1,"timeout state=%0d",dut.state);end
endmodule
