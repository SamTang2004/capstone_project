module softmax_int8 #(
 parameter int MAX_LENGTH=256,parameter int LENGTH_WIDTH=$clog2(MAX_LENGTH+1)
)(
 input logic clk,rst_n,start,input logic[LENGTH_WIDTH-1:0]vector_length,input logic input_valid,output_ready,
 output logic input_ready,input logic signed[7:0]input_data,output logic output_valid,output logic[7:0]output_probability_q7,
 output logic output_last,busy,done,error
);
 typedef enum logic[2:0]{IDLE,LOAD,EXP_SUM,DIV_START,DIV_WAIT,EMIT}state_t;state_t state;
 logic signed[7:0]scores[0:MAX_LENGTH-1];logic[15:0]exponentials[0:MAX_LENGTH-1];logic[LENGTH_WIDTH-1:0]length_r,index;
 logic signed[7:0]maximum;logic[31:0]exponential_sum,dividend,div_quotient,div_remainder;logic[15:0]current_exp;logic div_start,div_busy,div_done,div_zero;
 function automatic[15:0]exp_q15(input logic signed[8:0]delta);integer distance;begin distance=-delta;case(distance)
  0:exp_q15=32767;1:exp_q15=25520;2:exp_q15=19874;3:exp_q15=15480;4:exp_q15=12055;5:exp_q15=9388;6:exp_q15=7312;7:exp_q15=5695;8:exp_q15=4435;9:exp_q15=3454;10:exp_q15=2690;11:exp_q15=2095;12:exp_q15=1631;13:exp_q15=1270;14:exp_q15=989;15:exp_q15=770;16:exp_q15=600;default:exp_q15=0;endcase end endfunction
 iterative_unsigned_divider #(.WIDTH(32))divider(.clk,.rst_n,.start(div_start),.dividend,.divisor(exponential_sum),.busy(div_busy),.done(div_done),.divide_by_zero(div_zero),.quotient(div_quotient),.remainder(div_remainder));
 assign input_ready=state==LOAD;assign output_valid=state==EMIT;assign output_last=output_valid&&index==length_r-1'b1;assign busy=state!=IDLE;assign current_exp=exp_q15($signed(scores[index])-$signed(maximum));assign dividend=exponentials[index]*127+(exponential_sum>>1);assign div_start=state==DIV_START&&!div_busy;assign output_probability_q7=div_quotient>127?8'd127:div_quotient[7:0];
 always_ff@(posedge clk)begin if(!rst_n)begin state<=IDLE;length_r<=0;index<=0;maximum<=-8'sd128;exponential_sum<=0;done<=0;error<=0;end else begin done<=0;case(state)
  IDLE:if(start)begin error<=0;if(vector_length==0||vector_length>MAX_LENGTH)begin error<=1;done<=1;end else begin length_r<=vector_length;index<=0;maximum<=-8'sd128;state<=LOAD;end end
  LOAD:if(input_valid)begin scores[index]<=input_data;if(input_data>maximum)maximum<=input_data;if(index==length_r-1'b1)begin index<=0;exponential_sum<=0;state<=EXP_SUM;end else index<=index+1'b1;end
  EXP_SUM:begin exponentials[index]<=current_exp;exponential_sum<=exponential_sum+current_exp;if(index==length_r-1'b1)begin index<=0;state<=DIV_START;end else index<=index+1'b1;end
  DIV_START:if(div_start)state<=DIV_WAIT;DIV_WAIT:if(div_done)begin if(div_zero)error<=1;state<=EMIT;end
  EMIT:if(output_ready)begin if(index==length_r-1'b1)begin state<=IDLE;done<=1;end else begin index<=index+1'b1;state<=DIV_START;end end
  default:state<=IDLE;endcase end end
endmodule
