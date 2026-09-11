module iterative_unsigned_divider #(
    parameter int WIDTH = 32,
    parameter int COUNT_WIDTH = $clog2(WIDTH + 1)
) (
    input logic clk,rst_n,start,input logic [WIDTH-1:0] dividend,divisor,
    output logic busy,done,divide_by_zero,
    output logic [WIDTH-1:0] quotient,remainder
);
    logic [WIDTH-1:0] divisor_r,dividend_shift,quotient_work,quotient_next;
    logic [WIDTH:0] remainder_work,shifted_remainder;
    logic [COUNT_WIDTH-1:0] count;
    always_comb begin
        shifted_remainder={remainder_work[WIDTH-1:0],dividend_shift[WIDTH-1]};
        quotient_next={quotient_work[WIDTH-2:0],1'b0};
        if(shifted_remainder>={1'b0,divisor_r})quotient_next[0]=1'b1;
    end
    always_ff@(posedge clk)begin
      if(!rst_n)begin busy<=0;done<=0;divide_by_zero<=0;quotient<='0;remainder<='0;divisor_r<='0;dividend_shift<='0;quotient_work<='0;remainder_work<='0;count<='0;end
      else begin done<=0;
        if(start&&!busy)begin divide_by_zero<=divisor==0;if(divisor==0)begin quotient<='0;remainder<=dividend;done<=1;end else begin busy<=1;divisor_r<=divisor;dividend_shift<=dividend;quotient_work<='0;remainder_work<='0;count<=WIDTH;end end
        else if(busy)begin dividend_shift<={dividend_shift[WIDTH-2:0],1'b0};quotient_work<=quotient_next;if(shifted_remainder>={1'b0,divisor_r})remainder_work<=shifted_remainder-{1'b0,divisor_r};else remainder_work<=shifted_remainder;count<=count-1'b1;if(count==1)begin busy<=0;done<=1;quotient<=quotient_next;if(shifted_remainder>={1'b0,divisor_r})remainder<=shifted_remainder[WIDTH-1:0]-divisor_r;else remainder<=shifted_remainder[WIDTH-1:0];end end
      end
    end
endmodule
