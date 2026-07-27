`timescale 1ns/1ps
module systolic_array_8x8_int4 #(
    parameter bit USE_DSP = 1'b0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic                    enable,
    input  logic signed [4*8-1:0]   a_west_flat,
    input  logic signed [4*8-1:0]   b_north_flat,
    output logic signed [32*64-1:0] c_flat
);

    logic signed [3:0]  a_chain [0:7][0:7];
    logic signed [3:0]  b_chain [0:7][0:7];
    logic signed [31:0] acc     [0:7][0:7];

    genvar r;
    genvar c;

    generate
        for (r = 0; r < 8; r = r + 1) begin : gen_rows
            for (c = 0; c < 8; c = c + 1) begin : gen_cols
                pe_int4_mac #(.USE_DSP(USE_DSP)) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .clear(clear),
                    .enable(enable),
                    .a_in(c == 0 ? a_west_flat[r*4 +: 4] : a_chain[r][c-1]),
                    .b_in(r == 0 ? b_north_flat[c*4 +: 4] : b_chain[r-1][c]),
                    .a_out(a_chain[r][c]),
                    .b_out(b_chain[r][c]),
                    .acc_out(acc[r][c])
                );

                assign c_flat[(r*8 + c)*32 +: 32] = acc[r][c];
            end
        end
    endgenerate

endmodule
