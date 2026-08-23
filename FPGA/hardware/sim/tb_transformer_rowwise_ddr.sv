module tb_transformer_rowwise_ddr;
 logic clk=0,rst_n=0,pipeline_enable=1,start=0,softmax_mode=0,output_tiled=0;always#5 clk=~clk;
 logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall,rv,rr=1,rwide,rqword,rahead,dv=0,dr,dlast=0,wv,wr=1,wlast,memory_busy=0,memory_error=0;
 logic[31:0]raddr,waddr,wdata;logic[6:0]rlen;logic[7:0]rdata;logic[3:0]wstrb;logic[7:0]mem[0:2047];integer errors=0;
 transformer_rowwise_ddr dut(.clk,.rst_n,.pipeline_enable,.start,.softmax_mode,.output_tiled,.input_base_addr(32'h100),.gamma_base_addr(32'h200),.beta_base_addr(32'h240),.output_base_addr(32'h300),.epsilon_term(0),.rows(1),.columns(4),.busy,.done,.error,.op_start,.op_done,.memory_stall,.input_stall,.output_stall,.read_req_valid(rv),.read_req_ready(rr),.read_req_addr(raddr),.read_req_len(rlen),.read_req_wide(rwide),.read_req_qword(rqword),.read_req_allow_ahead(rahead),.read_data_valid(dv),.read_data_ready(dr),.read_data(rdata),.read_data_last(dlast),.write_req_valid(wv),.write_req_ready(wr),.write_req_addr(waddr),.write_req_data(wdata),.write_req_strb(wstrb),.write_req_last(wlast),.memory_busy,.memory_error);
 always_ff@(posedge clk)begin dv<=0;dlast<=0;if(rv&&rr)begin rdata<=mem[raddr];dv<=1;dlast<=1;end if(wv&&wr)for(int i=0;i<4;i++)if(wstrb[i])mem[waddr+i]<=wdata[i*8+:8];end
 task automatic launch(input logic mode,input logic tiled);begin softmax_mode=mode;output_tiled=tiled;@(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(posedge clk);if(error)errors++;end endtask
 initial begin for(int i=0;i<2048;i++)mem[i]=0;mem['h100]=8'hfd;mem['h101]=8'hff;mem['h102]=1;mem['h103]=3;for(int i=0;i<4;i++)begin mem['h200+i*2]=0;mem['h201+i*2]=1;end repeat(4)@(posedge clk);rst_n=1;
  launch(0,1);if(mem['h300]!==8'hfe||mem['h308]!==8'hff||mem['h310]!==1||mem['h318]!==2)errors++;
  for(int i=0;i<4;i++)mem['h100+i]=5;launch(1,0);for(int i=0;i<4;i++)if(mem['h300+i]!==32)errors++;
  if(errors)$fatal(1,"rowwise DDR failed %0d",errors);$display("Transformer LayerNorm/Softmax DDR golden trace passed");$finish;end
 initial begin#200000;$fatal(1,"timeout state=%0d",dut.state);end
endmodule
