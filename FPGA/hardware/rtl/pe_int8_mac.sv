module pe_int8_mac #(
    parameter bit USE_DSP = 1'b0
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               clear,
    input  logic               enable,
    input  logic signed [7:0]  a_in,
    input  logic signed [7:0]  b_in,
    output logic signed [7:0]  a_out,
    output logic signed [7:0]  b_out,
    output logic signed [31:0] acc_out
);

    logic signed [7:0] operand_a_pipe, operand_b_pipe;
    logic operand_valid;

    // Isolate the multiplier from the systolic routing and BRAM output path.
    // For DSP PEs these registers can be absorbed into the DSP48 A/B input
    // stages; LUT PEs receive the same latency and a much shorter input path.
    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            operand_a_pipe <= '0;
            operand_b_pipe <= '0;
            operand_valid <= 1'b0;
        end else begin
            if (enable) begin
                operand_a_pipe <= a_in;
                operand_b_pipe <= b_in;
            end
            operand_valid <= enable;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end

    generate
        if (USE_DSP) begin : dsp_mac
            // Instantiate the primitive so this PE is deterministically one
            // DSP48E1. OPMODE 0100101 selects P <- P + A*B; CEP stalls P when
            // there is no valid operand pair.
            logic signed [29:0] dsp_a;
            logic signed [17:0] dsp_b;
            logic signed [47:0] dsp_p;
            logic dsp_product_valid;

            assign dsp_a = {{22{operand_a_pipe[7]}}, operand_a_pipe};
            assign dsp_b = {{10{operand_b_pipe[7]}}, operand_b_pipe};
            assign acc_out = dsp_p[31:0];

            always_ff @(posedge clk) begin
                if (!rst_n || clear) dsp_product_valid <= 1'b0;
                else dsp_product_valid <= operand_valid;
            end

            DSP48E1 #(
                .A_INPUT("DIRECT"),
                .B_INPUT("DIRECT"),
                .USE_DPORT("FALSE"),
                .USE_MULT("MULTIPLY"),
                .USE_SIMD("ONE48"),
                .AREG(0),
                .BREG(0),
                .MREG(1),
                .PREG(1),
                .ACASCREG(0),
                .BCASCREG(0),
                .ADREG(0),
                .ALUMODEREG(0),
                .CARRYINREG(0),
                .CARRYINSELREG(0),
                .CREG(0),
                .DREG(0),
                .INMODEREG(0),
                .OPMODEREG(0)
            ) mac_dsp (
                .CLK(clk),
                .A(dsp_a),
                .B(dsp_b),
                .C(48'b0),
                .D(25'b0),
                .ACIN(30'b0),
                .BCIN(18'b0),
                .PCIN(48'b0),
                .CARRYCASCIN(1'b0),
                .CARRYIN(1'b0),
                .MULTSIGNIN(1'b0),
                .ALUMODE(4'b0000),
                .CARRYINSEL(3'b000),
                .INMODE(5'b00000),
                .OPMODE(7'b0100101),
                .CEA1(1'b0),
                .CEA2(1'b0),
                .CEAD(1'b0),
                .CEALUMODE(1'b0),
                .CEB1(1'b0),
                .CEB2(1'b0),
                .CEC(1'b0),
                .CECARRYIN(1'b0),
                .CECTRL(1'b0),
                .CED(1'b0),
                .CEINMODE(1'b0),
                .CEM(operand_valid),
                .CEP(dsp_product_valid),
                .RSTA(1'b0),
                .RSTALLCARRYIN(1'b0),
                .RSTALUMODE(1'b0),
                .RSTB(1'b0),
                .RSTC(1'b0),
                .RSTCTRL(1'b0),
                .RSTD(1'b0),
                .RSTINMODE(1'b0),
                .RSTM(!rst_n || clear),
                .RSTP(!rst_n || clear),
                .ACOUT(),
                .BCOUT(),
                .CARRYCASCOUT(),
                .CARRYOUT(),
                .MULTSIGNOUT(),
                .OVERFLOW(),
                .P(dsp_p),
                .PATTERNBDETECT(),
                .PATTERNDETECT(),
                .PCOUT(),
                .UNDERFLOW()
            );
        end else begin : lut_mac
            (* use_dsp = "no" *) logic signed [15:0] product_lut;
            logic product_valid;
            always_ff @(posedge clk) begin
                if (!rst_n || clear) product_lut <= '0;
                else if (operand_valid) product_lut <= operand_a_pipe * operand_b_pipe;
            end
            always_ff @(posedge clk) begin
                if (!rst_n || clear) begin
                    acc_out <= '0;
                    product_valid <= 1'b0;
                end else begin
                    if (product_valid) begin
                        acc_out <= acc_out + {{16{product_lut[15]}}, product_lut};
                    end
                    product_valid <= operand_valid;
                end
            end
        end
    endgenerate

endmodule
