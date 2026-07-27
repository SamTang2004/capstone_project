`timescale 1ns/1ps
module tb_activation_int4_parallel;
    logic enable;
    logic leaky_mode;
    logic [2:0] leaky_shift;
    logic signed [31:0] input_flat;
    logic signed [31:0] output_flat;

    activation_int4_parallel dut (.*);

    initial begin
        enable = 1'b1;
        leaky_mode = 1'b0;
        leaky_shift = 3'd2;
        input_flat = 32'h7650_1f98;
        #1;
        if (output_flat !== 32'h7650_1000)
            $fatal(1, "ReLU mismatch: %h", output_flat);

        leaky_mode = 1'b1;
        #1;
        if (output_flat !== 32'h7650_1fee)
            $fatal(1, "Leaky ReLU mismatch: %h", output_flat);

        enable = 1'b0;
        #1;
        if (output_flat !== input_flat)
            $fatal(1, "Disabled bypass mismatch");

        $display("tb_activation_int4_parallel golden trace passed");
        $finish;
    end
endmodule
