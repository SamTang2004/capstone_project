`timescale 1ns/1ps
module tb_requantize_int4_pipeline;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic input_valid;
    logic signed [31:0] input_value;
    logic signed [31:0] multiplier;
    logic [5:0] right_shift;
    logic signed [31:0] output_zero_point;
    logic output_valid;
    logic signed [3:0] output_value;
    integer sent;
    integer received;
    logic signed [3:0] expected [0:7];
    logic signed [31:0] stimulus [0:7];

    always #5 clk = ~clk;
    requantize_int4_pipeline dut (.*);

    initial begin
        stimulus[0] = -300; expected[0] = -8;
        stimulus[1] = -14;  expected[1] = -6;
        stimulus[2] = -2;   expected[2] = 0;
        stimulus[3] = 0;    expected[3] = 1;
        stimulus[4] = 8;    expected[4] = 5;
        stimulus[5] = 12;   expected[5] = 7;
        stimulus[6] = 500;  expected[6] = 7;
        stimulus[7] = -1;   expected[7] = 0;
        input_valid = 1'b0;
        input_value = '0;
        multiplier = 32'sd1;
        right_shift = 6'd1;
        output_zero_point = 32'sd1;
        sent = 0;
        received = 0;
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;

        while (received < 8) begin
            @(negedge clk);
            if (sent < 8) begin
                input_valid <= 1'b1;
                input_value <= stimulus[sent];
                sent = sent + 1;
            end else begin
                input_valid <= 1'b0;
            end
            @(posedge clk);
            #1;
            if (output_valid) begin
                if (output_value !== expected[received])
                    $fatal(1, "Output %0d mismatch: got %0d expected %0d",
                           received, output_value, expected[received]);
                received = received + 1;
            end
        end
        $display("tb_requantize_int4_pipeline golden trace passed");
        $finish;
    end
endmodule
