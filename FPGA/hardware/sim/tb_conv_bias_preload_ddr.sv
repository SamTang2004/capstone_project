module tb_conv_bias_preload_ddr;
    logic clk=0, rst_n=0, start, rv, rr, rwv, rwr, rwl, av, ar, busy, done, error;
    logic [31:0] ra, rwd; logic [4:0] rl; logic [3:0] rwk;
    logic [7:0] aa; logic signed [31:0] ad;
    logic [31:0] mem[0:3]; logic signed [31:0] acc[0:255];
    logic pending; logic [31:0] pending_addr; integer errors, requests;
    always #5 clk=~clk;
    assign rr=!pending&&!rwv; assign ar=1'b1;
    conv_bias_preload_ddr dut(
        .clk(clk),.rst_n(rst_n),.start(start),.bias_base_addr(0),.channel_offset(1),.tile_rows(2),.tile_cols(3),
        .busy(busy),.done(done),.error(error),.read_req_valid(rv),.read_req_ready(rr),.read_req_addr(ra),.read_req_len(rl),
        .read_word_valid(rwv),.read_word_ready(rwr),.read_word_data(rwd),.read_word_keep(rwk),.read_word_last(rwl),
        .accum_valid(av),.accum_ready(ar),.accum_addr(aa),.accum_data(ad));
    always @(posedge clk) begin
        if(rwv&&rwr)rwv<=0;
        if(pending&&!rwv)begin rwd<=mem[pending_addr>>2];rwk<=4'hf;rwl<=1;rwv<=1;pending<=0;end
        if(rv&&rr)begin if(rl!=4)$fatal(1,"Bias request length %0d",rl);pending<=1;pending_addr<=ra;requests<=requests+1;end
        if(av&&ar)acc[aa]<=ad;
    end
    initial begin
        errors=0;requests=0;start=0;rwv=0;rwd=0;rwk=0;rwl=0;pending=0;pending_addr=0;
        mem[1]=-7;mem[2]=13;mem[3]=-21;for(int i=0;i<256;i++)acc[i]=0;
        repeat(3)@(posedge clk);rst_n=1;@(negedge clk);start=1;@(negedge clk);start=0;wait(done);#1;
        if(acc[0]!=-7||acc[16]!=-7||acc[1]!=13||acc[17]!=13||acc[2]!=-21||acc[18]!=-21)errors++;
        if(error||errors||requests!=3)$fatal(1,"bias preload wide failed requests=%0d",requests);
        $display("Conv2D bias accumulator preload wide golden trace passed: 3 words");$finish;
    end
endmodule
