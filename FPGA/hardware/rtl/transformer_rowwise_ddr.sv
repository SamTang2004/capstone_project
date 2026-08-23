module transformer_rowwise_ddr #(
    parameter int MAX_COLUMNS=1024
)(
 input logic clk,rst_n,pipeline_enable,start,softmax_mode,output_tiled,
 input logic[31:0]input_base_addr,gamma_base_addr,beta_base_addr,output_base_addr,epsilon_term,
 input logic[15:0]rows,columns,
 output logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall,
 output logic read_req_valid,input logic read_req_ready,output logic[31:0]read_req_addr,output logic[6:0]read_req_len,
 output logic read_req_wide,read_req_qword,read_req_allow_ahead,input logic read_data_valid,output logic read_data_ready,input logic[7:0]read_data,input logic read_data_last,
 output logic write_req_valid,input logic write_req_ready,output logic[31:0]write_req_addr,output logic[31:0]write_req_data,output logic[3:0]write_req_strb,output logic write_req_last,
 input logic memory_busy,memory_error
);
 typedef enum logic[4:0]{IDLE,ROW_START,IN_REQ,IN_RECV,IN_FEED,G0_REQ,G0_RECV,G1_REQ,G1_RECV,B0_REQ,B0_RECV,B1_REQ,B1_RECV,OUTPUT,WRITE_ARM,WRITE_WAIT,FINISH,FAIL}state_t;
 state_t state;logic mode_r,tiled_r;logic[15:0]rows_r,cols_r,row_idx,col_idx;logic[31:0]eps_r;logic[7:0]sample_r,gamma_lo,beta_lo;logic signed[15:0]gamma_r,beta_r;
 logic ln_start,ln_input_valid,ln_input_ready,ln_param_valid,ln_param_ready,ln_output_valid,ln_output_last,ln_busy,ln_done,ln_error;logic signed[7:0]ln_output_data;
 logic sm_start,sm_input_valid,sm_input_ready,sm_output_valid,sm_output_last,sm_busy,sm_done,sm_error;logic[7:0]sm_output_data;
 logic[31:0]byte_output_addr,input_addr_r,gamma_addr_r,beta_addr_r,output_row_base_r;logic[7:0]output_byte;
 layernorm_int8 #(.MAX_LENGTH(MAX_COLUMNS)) ln(.clk,.rst_n,.start(ln_start),.vector_length(cols_r),.epsilon_term(eps_r),.input_valid(ln_input_valid),.input_ready(ln_input_ready),.input_data(sample_r),.param_valid(ln_param_valid),.param_ready(ln_param_ready),.gamma_q8(gamma_r),.beta_q8(beta_r),.output_valid(ln_output_valid),.output_ready(write_req_ready&&state==OUTPUT),.output_data(ln_output_data),.output_last(ln_output_last),.busy(ln_busy),.done(ln_done),.error(ln_error));
 softmax_int8 sm(.clk,.rst_n,.start(sm_start),.vector_length(cols_r[8:0]),.input_valid(sm_input_valid),.input_ready(sm_input_ready),.input_data(sample_r),.output_valid(sm_output_valid),.output_ready(write_req_ready&&state==OUTPUT),.output_probability_q7(sm_output_data),.output_last(sm_output_last),.busy(sm_busy),.done(sm_done),.error(sm_error));
 assign ln_start=state==ROW_START&&!mode_r;assign sm_start=state==ROW_START&&mode_r;
 assign ln_input_valid=state==IN_FEED&&!mode_r;assign sm_input_valid=state==IN_FEED&&mode_r;assign ln_param_valid=state==OUTPUT&&!mode_r;
 assign busy=state!=IDLE&&state!=FINISH&&state!=FAIL;assign op_start=state==IDLE&&start;assign op_done=state==FINISH;
 assign read_req_valid=pipeline_enable&&(state==IN_REQ||state==G0_REQ||state==G1_REQ||state==B0_REQ||state==B1_REQ);assign read_req_len=1;assign read_req_wide=0;assign read_req_qword=0;assign read_req_allow_ahead=0;assign read_data_ready=pipeline_enable&&(state==IN_RECV||state==G0_RECV||state==G1_RECV||state==B0_RECV||state==B1_RECV);
 always_comb begin case(state)IN_REQ:read_req_addr=input_addr_r;G0_REQ:read_req_addr=gamma_addr_r;G1_REQ:read_req_addr=gamma_addr_r+1;B0_REQ:read_req_addr=beta_addr_r;default:read_req_addr=beta_addr_r+1;endcase end
 assign output_byte=mode_r?sm_output_data:ln_output_data;
 assign write_req_valid=pipeline_enable&&state==OUTPUT&&(mode_r?sm_output_valid:ln_output_valid);assign write_req_addr={byte_output_addr[31:2],2'b0};assign write_req_data={4{output_byte}}<<(byte_output_addr[1:0]*8);assign write_req_strb=4'b1<<byte_output_addr[1:0];assign write_req_last=1;
 assign memory_stall=busy&&((read_req_valid&&!read_req_ready)||(read_data_ready&&!read_data_valid)||(write_req_valid&&!write_req_ready));assign input_stall=read_data_ready&&!read_data_valid;assign output_stall=write_req_valid&&!write_req_ready;
 always_ff@(posedge clk)begin if(!rst_n)begin state<=IDLE;done<=0;error<=0;mode_r<=0;tiled_r<=0;rows_r<=0;cols_r<=0;row_idx<=0;col_idx<=0;eps_r<=0;sample_r<=0;gamma_lo<=0;beta_lo<=0;gamma_r<=0;beta_r<=0;input_addr_r<=0;gamma_addr_r<=0;beta_addr_r<=0;output_row_base_r<=0;byte_output_addr<=0;end else begin done<=0;if(memory_error||ln_error||sm_error)begin error<=1;state<=FAIL;end else if(pipeline_enable)unique case(state)
  IDLE:if(start)begin error<=0;if(rows==0||columns==0||columns>MAX_COLUMNS||(softmax_mode&&columns>256))begin error<=1;state<=FAIL;end else begin mode_r<=softmax_mode;tiled_r<=output_tiled;rows_r<=rows;cols_r<=columns;eps_r<=epsilon_term;row_idx<=0;col_idx<=0;input_addr_r<=input_base_addr;output_row_base_r<=output_base_addr;state<=ROW_START;end end
  ROW_START:begin col_idx<=0;gamma_addr_r<=gamma_base_addr;beta_addr_r<=beta_base_addr;byte_output_addr<=output_row_base_r;state<=IN_REQ;end IN_REQ:if(read_req_valid&&read_req_ready)state<=IN_RECV;IN_RECV:if(read_data_valid)begin if(!read_data_last)begin error<=1;state<=FAIL;end else begin sample_r<=read_data;state<=IN_FEED;end end
  IN_FEED:if(mode_r?sm_input_ready:ln_input_ready)begin input_addr_r<=input_addr_r+1;if(col_idx+1==cols_r)begin col_idx<=0;state<=mode_r?OUTPUT:G0_REQ;end else begin col_idx<=col_idx+1;state<=IN_REQ;end end
  G0_REQ:if(read_req_valid&&read_req_ready)state<=G0_RECV;G0_RECV:if(read_data_valid)begin gamma_lo<=read_data;state<=G1_REQ;end G1_REQ:if(read_req_valid&&read_req_ready)state<=G1_RECV;G1_RECV:if(read_data_valid)begin gamma_r<={read_data,gamma_lo};state<=B0_REQ;end
  B0_REQ:if(read_req_valid&&read_req_ready)state<=B0_RECV;B0_RECV:if(read_data_valid)begin beta_lo<=read_data;state<=B1_REQ;end B1_REQ:if(read_req_valid&&read_req_ready)state<=B1_RECV;B1_RECV:if(read_data_valid)begin beta_r<={read_data,beta_lo};state<=OUTPUT;end
  OUTPUT:if(write_req_valid&&write_req_ready)begin if((mode_r?sm_output_last:ln_output_last))begin if(row_idx+1==rows_r)state<=WRITE_ARM;else begin row_idx<=row_idx+1;if(tiled_r)begin if(row_idx[2:0]==3'd7)output_row_base_r<=output_row_base_r+(cols_r<<3)-7;else output_row_base_r<=output_row_base_r+1;end else output_row_base_r<=byte_output_addr+1;state<=ROW_START;end end else begin col_idx<=col_idx+1;gamma_addr_r<=gamma_addr_r+2;beta_addr_r<=beta_addr_r+2;byte_output_addr<=byte_output_addr+(tiled_r?8:1);state<=mode_r?OUTPUT:G0_REQ;end end
  WRITE_ARM:state<=WRITE_WAIT;WRITE_WAIT:if(!memory_busy)state<=FINISH;FINISH:begin done<=1;state<=IDLE;end FAIL:begin done<=1;if(!start)state<=IDLE;end default:state<=IDLE;endcase end end
endmodule
