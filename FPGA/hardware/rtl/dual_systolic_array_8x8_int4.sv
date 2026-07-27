`timescale 1ns/1ps
module dual_systolic_array_8x8_int4 #(
    parameter bit USE_DSP = 1'b0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic                    enable,
    input  logic [1:0]              share_mode,
    input  logic signed [4*8-1:0]   a0_west_flat,
    input  logic signed [4*8-1:0]   b0_north_flat,
    input  logic signed [4*8-1:0]   a1_west_flat,
    input  logic signed [4*8-1:0]   b1_north_flat,
    output logic signed [32*64-1:0] c0_flat,
    output logic signed [32*64-1:0] c1_flat
);

    localparam logic [1:0] SHARE_NONE = 2'b00;
    localparam logic [1:0] SHARE_A    = 2'b01;
    localparam logic [1:0] SHARE_B    = 2'b10;

    logic signed [4*8-1:0] array1_a;
    logic signed [4*8-1:0] array1_b;

    always_comb begin
        array1_a = a1_west_flat;
        array1_b = b1_north_flat;

        case (share_mode)
            SHARE_A: array1_a = a0_west_flat;
            SHARE_B: array1_b = b0_north_flat;
            default: begin end
        endcase
    end

    systolic_array_8x8_int4 #(.USE_DSP(USE_DSP)) array0 (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_west_flat(a0_west_flat),
        .b_north_flat(b0_north_flat),
        .c_flat(c0_flat)
    );

    systolic_array_8x8_int4 #(.USE_DSP(USE_DSP)) array1 (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_west_flat(array1_a),
        .b_north_flat(array1_b),
        .c_flat(c1_flat)
    );

endmodule
