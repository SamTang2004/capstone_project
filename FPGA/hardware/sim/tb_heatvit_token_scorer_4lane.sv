`timescale 1ns/1ps
module tb_heatvit_token_scorer_4lane;
    logic clk = 0, rst_n = 0, start_token, feature_valid, feature_last;
    logic signed [7:0] activation [0:3], weight [0:3];
    logic signed [31:0] bias;
    logic feature_ready, score_valid;
    logic signed [39:0] score;
    always #5 clk = ~clk;

    heatvit_token_scorer_4lane dut (.*);

    initial begin
        start_token = 0; feature_valid = 0; feature_last = 0; bias = 5;
        repeat (2) @(posedge clk); rst_n <= 1;
        @(negedge clk); start_token = 1;
        @(negedge clk); start_token = 0; feature_valid = 1;
        activation[0] = 1; activation[1] = 2; activation[2] = 3; activation[3] = 4;
        weight[0] = 2; weight[1] = 2; weight[2] = 2; weight[3] = 2;
        @(negedge clk);
        activation[0] = -1; activation[1] = 1; activation[2] = -2; activation[3] = 2;
        feature_last = 1;
        @(negedge clk); feature_valid = 0; feature_last = 0;
        @(negedge clk);
        if (!score_valid || score !== 25) $fatal(1, "score mismatch: %0d", score);
        $display("HeatViT four-lane scorer golden trace passed: score=%0d", score);
        $finish;
    end
endmodule
