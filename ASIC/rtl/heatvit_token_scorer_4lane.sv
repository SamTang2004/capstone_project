`timescale 1ns/1ps
module heatvit_token_scorer_4lane #(
    parameter int ACC_WIDTH = 40
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       start_token,
    input  logic                       feature_valid,
    input  logic                       feature_last,
    input  logic signed [7:0]          activation [0:3],
    input  logic signed [7:0]          weight [0:3],
    input  logic signed [31:0]         bias,
    output logic                       feature_ready,
    output logic                       score_valid,
    output logic signed [ACC_WIDTH-1:0] score
);
    logic signed [ACC_WIDTH-1:0] accumulator;
    (* use_dsp = "yes" *) logic signed [15:0] product_0, product_1, product_2, product_3;
    logic product_valid, product_last;
    logic signed [17:0] product_sum;

    /* Register each product before the reduction tree.  Besides mapping the
     * four lanes to DSP48E1s, this removes the weight-cache mux and multiplier
     * tree from the accumulator's critical path. */
    assign product_sum = $signed(product_0) + $signed(product_1) +
                         $signed(product_2) + $signed(product_3);

    assign feature_ready = 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            accumulator <= '0;
            score <= '0;
            score_valid <= 1'b0;
            product_0 <= '0; product_1 <= '0; product_2 <= '0; product_3 <= '0;
            product_valid <= 1'b0; product_last <= 1'b0;
        end else begin
            score_valid <= 1'b0;
            product_valid <= feature_valid;
            product_last <= feature_last;
            if (feature_valid) begin
                product_0 <= $signed(activation[0]) * $signed(weight[0]);
                product_1 <= $signed(activation[1]) * $signed(weight[1]);
                product_2 <= $signed(activation[2]) * $signed(weight[2]);
                product_3 <= $signed(activation[3]) * $signed(weight[3]);
            end
            if (start_token)
                accumulator <= {{(ACC_WIDTH-32){bias[31]}}, bias};
            if (product_valid) begin
                if (product_last) begin
                    score <= accumulator + product_sum;
                    score_valid <= 1'b1;
                    accumulator <= '0;
                end else begin
                    accumulator <= accumulator + product_sum;
                end
            end
        end
    end
endmodule
