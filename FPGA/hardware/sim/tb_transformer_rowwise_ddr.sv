module tb_transformer_rowwise_ddr;
 logic clk=0,rst_n=0,pipeline_enable=1,start=0,softmax_mode=0,output_tiled=0;always#5 clk=~clk;
 logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall,rv,rr=1,rwide,rqword,rahead,dv=0,dr,dlast=0,wv,wr=1,wlast,memory_busy=0,memory_error=0;
 logic[31:0]raddr,waddr,wdata;logic[6:0]rlen;logic[7:0]rdata;logic[3:0]wstrb;logic[7:0]mem[0:2047];integer errors=0,read_requests=0,write_requests=0,write_lasts=0;
 logic[15:0] rows_cfg=1,cols_cfg=4;
 logic read_pending=0;logic[31:0]read_base=0;logic[6:0]read_count=0,read_length=0;
 transformer_rowwise_ddr dut(.clk,.rst_n,.pipeline_enable,.start,.softmax_mode,.output_tiled,.input_base_addr(32'h100),.gamma_base_addr(32'h200),.beta_base_addr(32'h240),.output_base_addr(32'h300),.epsilon_term(0),.rows(rows_cfg),.columns(cols_cfg),.busy,.done,.error,.op_start,.op_done,.memory_stall,.input_stall,.output_stall,.read_req_valid(rv),.read_req_ready(rr),.read_req_addr(raddr),.read_req_len(rlen),.read_req_wide(rwide),.read_req_qword(rqword),.read_req_allow_ahead(rahead),.read_data_valid(dv),.read_data_ready(dr),.read_data(rdata),.read_data_last(dlast),.write_req_valid(wv),.write_req_ready(wr),.write_req_addr(waddr),.write_req_data(wdata),.write_req_strb(wstrb),.write_req_last(wlast),.memory_busy,.memory_error);
 always_ff@(posedge clk)begin
  dv<=0;dlast<=0;
  if(rv&&rr)begin read_pending<=1;read_base<=raddr;read_count<=0;read_length<=rlen;read_requests<=read_requests+1;end
  if(read_pending)begin
   dv<=1;rdata<=mem[read_base+read_count];dlast<=(read_count+1==read_length);
   if(dr)begin
    if(read_count+1==read_length)read_pending<=0;
    else read_count<=read_count+1;
   end
  end
  if(wv&&wr)begin write_requests<=write_requests+1;if(wlast)write_lasts<=write_lasts+1;for(int i=0;i<4;i++)if(wstrb[i])mem[waddr+i]<=wdata[i*8+:8];end
 end
 task automatic launch(input logic mode,input logic tiled);begin softmax_mode=mode;output_tiled=tiled;@(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(posedge clk);if(error)errors++;end endtask
 initial begin for(int i=0;i<2048;i++)mem[i]=0;mem['h100]=8'hfd;mem['h101]=8'hff;mem['h102]=1;mem['h103]=3;for(int i=0;i<4;i++)begin mem['h200+i*2]=0;mem['h201+i*2]=i+1;end repeat(4)@(posedge clk);rst_n=1;
  launch(0,1);if(mem['h300]!==8'hfe||mem['h308]!==8'hfe||mem['h310]!==2||mem['h318]!==6)begin $display("LayerNorm output got %0d %0d %0d %0d gamma=%h,%h,%h,%h beta=%h,%h,%h,%h",$signed(mem['h300]),$signed(mem['h308]),$signed(mem['h310]),$signed(mem['h318]),dut.gamma_mem[0],dut.gamma_mem[1],dut.gamma_mem[2],dut.gamma_mem[3],dut.beta_mem[0],dut.beta_mem[1],dut.beta_mem[2],dut.beta_mem[3]);errors++;end if(read_requests!==3)begin $display("LayerNorm expected 3 burst reads, got %0d",read_requests);errors++;end
  for(int i=0;i<4;i++)mem['h100+i]=5;launch(1,0);for(int i=0;i<4;i++)if(mem['h300+i]!==32)errors++;if(read_requests!==4)begin $display("Softmax expected one additional burst read, got %0d",read_requests);errors++;end
  launch(1,1);if(mem['h300]!==32||mem['h308]!==32||mem['h310]!==32||mem['h318]!==32)errors++;if(read_requests!==5)begin $display("Tiled Softmax expected one additional burst read, got %0d",read_requests);errors++;end
  begin integer writes_before,lasts_before;writes_before=write_requests;lasts_before=write_lasts;rows_cfg=8;for(int i=0;i<32;i++)mem['h100+i]=5;launch(1,1);for(int col=0;col<4;col++)for(int row=0;row<8;row++)if(mem['h300+col*8+row]!==32)errors++;if(write_requests-writes_before!==8||write_lasts-lasts_before!==1)begin $display("Tiled 8x4 Softmax expected 8 words/1 burst, got %0d/%0d",write_requests-writes_before,write_lasts-lasts_before);errors++;end end
  if(errors)$fatal(1,"rowwise DDR failed %0d",errors);$display("Transformer LayerNorm/Softmax DDR golden trace passed");$finish;end
 initial begin#200000;$fatal(1,"timeout state=%0d",dut.state);end
endmodule
