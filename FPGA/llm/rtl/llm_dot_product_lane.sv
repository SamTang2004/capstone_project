// Signed W8/A8 multiplier: A1/B1 -> A2/B2 -> M -> P (four clocks).
// enable qualifies input capture only; downstream stages always drain bubbles.
module llm_dot_product_lane (
    input logic clk, rst_n, enable,
    input logic signed [7:0] a, b,
    output wire signed [15:0] product
);
    wire [47:0] p;
    assign product = p[15:0];
    DSP48E1 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("FALSE"),
        .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"),
        .AREG(2), .BREG(2), .MREG(1), .PREG(1), .ACASCREG(2), .BCASCREG(2),
        .ADREG(0), .ALUMODEREG(1), .CARRYINREG(1), .CARRYINSELREG(1),
        .CREG(0), .DREG(0), .INMODEREG(1), .OPMODEREG(1)
    ) mul (
        .CLK(clk), .A({{22{a[7]}}, a}), .B({{10{b[7]}}, b}),
        .C(48'b0), .D(25'b0), .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .CARRYIN(1'b0), .MULTSIGNIN(1'b0),
        .ALUMODE(4'b0), .CARRYINSEL(3'b0), .INMODE(5'b0), .OPMODE(7'b0000101),
        .CEA1(enable), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1),
        .CEB1(enable), .CEB2(1'b1), .CEC(1'b0), .CECARRYIN(1'b1),
        .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),
        .RSTA(!rst_n), .RSTALLCARRYIN(!rst_n), .RSTALUMODE(!rst_n), .RSTB(!rst_n),
        .RSTC(1'b0), .RSTCTRL(!rst_n), .RSTD(1'b0), .RSTINMODE(!rst_n),
        .RSTM(!rst_n), .RSTP(!rst_n), .P(p), .ACOUT(), .BCOUT(), .PCOUT(),
        .CARRYCASCOUT(), .CARRYOUT(), .MULTSIGNOUT(), .OVERFLOW(),
        .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW()
    );
endmodule
