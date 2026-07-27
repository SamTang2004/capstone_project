`timescale 1ns/1ps
module pe_int4_mac #(
    parameter bit USE_DSP = 1'b0
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               clear,
    input  logic               enable,
    input  logic signed [3:0]  a_in,
    input  logic signed [3:0]  b_in,
    output logic signed [3:0]  a_out,
    output logic signed [3:0]  b_out,
    output logic signed [31:0] acc_out
);

    logic signed [3:0] operand_a_pipe;
    logic signed [3:0] operand_b_pipe;
    logic signed [7:0] product_pipe;
    logic operand_valid;
    logic product_valid;

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            operand_a_pipe <= '0;
            operand_b_pipe <= '0;
            operand_valid <= 1'b0;
            a_out <= '0;
            b_out <= '0;
        end else begin
            operand_valid <= enable;
            if (enable) begin
                operand_a_pipe <= a_in;
                operand_b_pipe <= b_in;
                a_out <= a_in;
                b_out <= b_in;
            end else begin
                a_out <= '0;
                b_out <= '0;
            end
        end
    end

    generate
        if (USE_DSP) begin : dsp_multiplier
            (* use_dsp = "yes" *) logic signed [7:0] product_dsp;
            always_ff @(posedge clk) begin
                if (!rst_n || clear)
                    product_dsp <= '0;
                else if (operand_valid)
                    product_dsp <= operand_a_pipe * operand_b_pipe;
            end
            assign product_pipe = product_dsp;
        end else begin : lut_multiplier
            (* use_dsp = "no" *) logic signed [7:0] product_lut;
            always_ff @(posedge clk) begin
                if (!rst_n || clear)
                    product_lut <= '0;
                else if (operand_valid)
                    product_lut <= operand_a_pipe * operand_b_pipe;
            end
            assign product_pipe = product_lut;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            acc_out <= '0;
            product_valid <= 1'b0;
        end else begin
            if (product_valid)
                acc_out <= acc_out + {{24{product_pipe[7]}}, product_pipe};
            product_valid <= operand_valid;
        end
    end

endmodule
