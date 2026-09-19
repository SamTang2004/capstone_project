// Exhaustive signed products with bubbles and reset at every pipeline depth.
module tb_dsp_pipeline;
    logic clk=0, rst_n=0, enable=0;
    logic signed [7:0] a=0, b=0;
    wire signed [15:0] product;
    logic [3:0] valid_q=0;
    logic signed [15:0] expected [0:3];
    integer checked=0;
    llm_dot_product_lane dut(.*);
    always #5 clk=~clk;
    task automatic tick;
        @(posedge clk);
        if (!rst_n) valid_q=0;
        else begin
            for (integer i=3;i>0;i--) expected[i]=expected[i-1];
            expected[0]=$signed(a)*$signed(b);
            valid_q={valid_q[2:0],enable};
        end
        #1;
        if (valid_q[3]) begin
            if (product !== expected[3])
                $fatal(1,"DSP pipeline mismatch got=%0d expected=%0d",product,expected[3]);
            checked++;
        end
        @(negedge clk);
    endtask
    initial begin
        repeat(20) tick(); rst_n=1;
        for (integer x=-128;x<128;x++) begin
            for (integer y=-128;y<128;y++) begin
                a=x; b=y; enable=1; tick();
                if ((x+y)%7==0) begin
                    enable=0; a=99; b=-117; tick();
                end
            end
        end
        enable=0; repeat(4) tick();
        if (checked != 65536) $fatal(1,"Exhaustive product count");
        for (integer depth=1;depth<=4;depth++) begin
            enable=1; a=-128; b=-128; tick();
            enable=0; repeat(depth-1) tick();
            rst_n=0; tick(); rst_n=1;
            // Immediate post-reset traffic also checks registered static controls.
            enable=1; a=-128; b=127; tick();
            enable=0; repeat(4) tick();
        end
        $display("PASS: DSP pipeline exhaustive=65536 bubbles reset_depths=4 checked=%0d",checked);
        $finish;
    end
endmodule
