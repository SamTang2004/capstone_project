`timescale 1ns/1ps
module activation_int4_parallel #(
    parameter int LANES = 8
) (
    input  logic                         enable,
    input  logic                         leaky_mode,
    input  logic [2:0]                   leaky_shift,
    input  logic signed [LANES*4-1:0]    input_flat,
    output logic signed [LANES*4-1:0]    output_flat
);
    always_comb begin
        output_flat = input_flat;
        if (enable) begin
            for (int lane = 0; lane < LANES; lane++) begin
                if ($signed(input_flat[lane*4 +: 4]) < 0) begin
                    output_flat[lane*4 +: 4] = leaky_mode
                        ? $signed(input_flat[lane*4 +: 4]) >>> leaky_shift
                        : 4'sd0;
                end
            end
        end
    end
endmodule
