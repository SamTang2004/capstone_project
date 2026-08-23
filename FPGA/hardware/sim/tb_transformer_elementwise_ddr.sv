module tb_transformer_elementwise_ddr;
 logic clk=0,rst_n=0,pipeline_enable=1,start=0,residual_mode=0;always#5 clk=~clk;
 logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
 logic rv,rr=1,rwide,rqword,rahead,qv=0,qr,qlast=0,wv,wr=1,wlast,memory_busy=0,memory_error=0;
 logic[31:0]raddr,waddr,wdata;logic[6:0]rlen;logic[63:0]qdata;logic[7:0]qkeep;logic[3:0]wstrb;logic[7:0]mem[0:1023];integer errors=0;
 transformer_elementwise_ddr dut(.clk,.rst_n,.pipeline_enable,.start,.residual_mode,
  .input0_base_addr(32'h100),.input1_base_addr(32'h200),.output_base_addr(32'h300),.total_bytes(32'd9),
  .lhs_right_shift(0),.rhs_right_shift(0),.busy,.done,.error,.op_start,.op_done,.memory_stall,.input_stall,.output_stall,
  .read_req_valid(rv),.read_req_ready(rr),.read_req_addr(raddr),.read_req_len(rlen),.read_req_wide(rwide),.read_req_qword(rqword),.read_req_allow_ahead(rahead),
  .read_qword_valid(qv),.read_qword_ready(qr),.read_qword_data(qdata),.read_qword_keep(qkeep),.read_qword_last(qlast),
  .write_req_valid(wv),.write_req_ready(wr),.write_req_addr(waddr),.write_req_data(wdata),.write_req_strb(wstrb),.write_req_last(wlast),.memory_busy,.memory_error);
 always_ff@(posedge clk)begin qv<=0;qlast<=0;if(rv&&rr)begin qdata<=0;qkeep<=0;for(int i=0;i<8;i++)if(i<rlen)begin qdata[i*8+:8]<=mem[raddr+i];qkeep[i]<=1;end qv<=1;qlast<=1;end if(wv&&wr)for(int i=0;i<4;i++)if(wstrb[i])mem[waddr+i]<=wdata[i*8+:8];end
 task automatic launch(input logic mode);begin residual_mode=mode;@(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(posedge clk);if(error)errors++;end endtask
 initial begin for(int i=0;i<1024;i++)mem[i]=0;for(int i=0;i<9;i++)begin mem['h100+i]=i==0?8'hf8:i;mem['h200+i]=10;end repeat(4)@(posedge clk);rst_n=1;
  launch(0);if(mem['h300]!==8'hfd||mem['h301]!==1||mem['h308]!==5)errors++;
  launch(1);if(mem['h300]!==2||mem['h301]!==11||mem['h308]!==18)errors++;
  if(errors)$fatal(1,"elementwise DDR failed %0d",errors);$display("Transformer GELU/residual DDR golden trace passed");$finish;end
 initial begin#100000;$fatal(1,"timeout");end
endmodule
