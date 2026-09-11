module layernorm_int8 #(
 parameter int MAX_LENGTH=1024,parameter int LENGTH_WIDTH=$clog2(MAX_LENGTH+1)
)(
 input logic clk,rst_n,start,input logic[LENGTH_WIDTH-1:0]vector_length,input logic[31:0]epsilon_term,
 input logic input_valid,output_ready,param_valid,input logic signed[7:0]input_data,
 output logic input_ready,param_ready,output_valid,input logic signed[15:0]gamma_q8,beta_q8,
 output logic signed[7:0]output_data,output logic output_last,busy,done,error
);
 typedef enum logic[4:0]{IDLE,LOAD,STATS,VAR_SUB,VAR_EPS,SQRT_TRIAL,SQRT_COMPARE,SQRT_SUB_LOW,SQRT_UPDATE,CENTER_MULT,CENTER_SUB,ABS_SHIFT,DIV_START,DIV_WAIT,SIGN_RESULT,MULTIPLY,SCALE,ROUND,EMIT}state_t;state_t state;
 logic signed[7:0]samples[0:MAX_LENGTH-1];logic[LENGTH_WIDTH-1:0]length_r,index;
 logic signed[39:0]sum;logic[47:0]sum_squares;logic signed[15:0]input_square;
 logic[63:0]sqrt_op,sqrt_res,sqrt_one,sqrt_trial,variance_product_a,variance_product_b,variance_hold;logic sqrt_take,sqrt_low_borrow;logic[31:0]sqrt_low_difference;
 logic[5:0]sqrt_count;logic[31:0]denominator;logic signed[63:0]norm_q14,centered_hold;logic signed[39:0]sample_product_hold;
 logic signed[63:0]norm_hold,scaled_hold,rounded_calc;logic signed[79:0]product_hold;logic signed[7:0]output_hold;
 logic numerator_negative;logic signed[15:0]gamma_hold,beta_hold;logic div_start,div_busy,div_done,div_zero;logic[63:0]div_quotient,div_remainder,abs_numerator_hold;
 iterative_unsigned_divider #(.WIDTH(64))divider(.clk,.rst_n,.start(div_start),.dividend(abs_numerator_hold),.divisor({32'b0,denominator}),.busy(div_busy),.done(div_done),.divide_by_zero(div_zero),.quotient(div_quotient),.remainder(div_remainder));
 always_comb begin
  input_square=input_data*input_data;
  norm_q14=numerator_negative?-$signed(div_quotient):$signed(div_quotient);rounded_calc=scaled_hold>=0?scaled_hold+128:scaled_hold-128;
 end
 assign input_ready=state==LOAD;assign param_ready=state==DIV_START&&!div_busy;assign div_start=param_ready&&param_valid;assign output_valid=state==EMIT;assign output_data=output_hold;assign output_last=output_valid&&index==length_r-1'b1;assign busy=state!=IDLE;
 always_ff@(posedge clk)begin
  if(!rst_n)begin state<=IDLE;length_r<='0;index<='0;sum<='0;sum_squares<='0;sqrt_op<='0;sqrt_res<='0;sqrt_one<='0;sqrt_trial<='0;variance_product_a<='0;variance_product_b<='0;variance_hold<='0;sqrt_take<=0;sqrt_low_borrow<=0;sqrt_low_difference<='0;sqrt_count<='0;denominator<='0;numerator_negative<=0;sample_product_hold<='0;centered_hold<='0;abs_numerator_hold<='0;gamma_hold<='0;beta_hold<='0;norm_hold<='0;product_hold<='0;scaled_hold<='0;output_hold<='0;done<=0;error<=0;end
  else begin done<=0;case(state)
   IDLE:if(start)begin error<=0;if(vector_length==0||vector_length>MAX_LENGTH)begin error<=1;done<=1;end else begin length_r<=vector_length;index<=0;sum<=0;sum_squares<=0;state<=LOAD;end end
   LOAD:if(input_valid)begin samples[index]<=input_data;sum<=sum+input_data;sum_squares<=sum_squares+$unsigned(input_square);if(index==length_r-1'b1)begin index<=0;state<=STATS;end else index<=index+1'b1;end
   STATS:begin variance_product_a<=length_r*sum_squares;variance_product_b<=sum*sum;state<=VAR_SUB;end
   VAR_SUB:begin variance_hold<=variance_product_a-variance_product_b;state<=VAR_EPS;end
   VAR_EPS:begin sqrt_op<=variance_hold+epsilon_term;sqrt_res<=0;sqrt_one<=64'h4000_0000_0000_0000;sqrt_count<=0;state<=SQRT_TRIAL;end
   SQRT_TRIAL:begin sqrt_trial<=sqrt_res+sqrt_one;state<=SQRT_COMPARE;end
   SQRT_COMPARE:begin sqrt_take<=sqrt_op>=sqrt_trial;state<=SQRT_SUB_LOW;end
   SQRT_SUB_LOW:begin sqrt_low_difference<=sqrt_op[31:0]-sqrt_trial[31:0];sqrt_low_borrow<=sqrt_op[31:0]<sqrt_trial[31:0];state<=SQRT_UPDATE;end
   SQRT_UPDATE:begin
    if(sqrt_take)begin sqrt_op<={sqrt_op[63:32]-sqrt_trial[63:32]-sqrt_low_borrow,sqrt_low_difference};sqrt_res<=(sqrt_res>>1)+sqrt_one;if(sqrt_count==31)denominator<=((sqrt_res>>1)+sqrt_one);end
    else begin sqrt_res<=sqrt_res>>1;if(sqrt_count==31)denominator<=sqrt_res>>1;end
    sqrt_one<=sqrt_one>>2;if(sqrt_count==31)begin index<=0;state<=CENTER_MULT;end else begin sqrt_count<=sqrt_count+1'b1;state<=SQRT_TRIAL;end
   end
   CENTER_MULT:begin sample_product_hold<=$signed(samples[index])*$signed({1'b0,length_r});state<=CENTER_SUB;end
   CENTER_SUB:begin centered_hold<=sample_product_hold-sum;state<=ABS_SHIFT;end
   ABS_SHIFT:begin numerator_negative<=centered_hold<0;abs_numerator_hold<=centered_hold<0?-(centered_hold<<<14):(centered_hold<<<14);state<=DIV_START;end
   DIV_START:if(div_start)begin gamma_hold<=gamma_q8;beta_hold<=beta_q8;state<=DIV_WAIT;end
   DIV_WAIT:if(div_done)begin if(div_zero)error<=1;state<=SIGN_RESULT;end
   SIGN_RESULT:begin norm_hold<=norm_q14;state<=MULTIPLY;end
   MULTIPLY:begin product_hold<=norm_hold*gamma_hold;state<=SCALE;end
   SCALE:begin scaled_hold<=(product_hold>>>14)+beta_hold;state<=ROUND;end
   ROUND:begin if((rounded_calc>>>8)>127)output_hold<=8'sd127;else if((rounded_calc>>>8)<-128)output_hold<=-8'sd128;else output_hold<=rounded_calc[15:8];state<=EMIT;end
   EMIT:if(output_ready)begin if(index==length_r-1'b1)begin state<=IDLE;done<=1;end else begin index<=index+1'b1;state<=CENTER_MULT;end end
   default:state<=IDLE;endcase end
 end
endmodule
