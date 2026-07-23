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
    logic signed [15:0] product_pipe;
    logic operand_valid, product_valid;

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

    generate
        if (USE_DSP) begin : dsp_multiplier
            (* use_dsp = "yes" *) logic signed [15:0] product_dsp;
            always_ff @(posedge clk) begin
                if (!rst_n || clear) product_dsp <= '0;
                else if (operand_valid) product_dsp <= operand_a_pipe * operand_b_pipe;
            end
            assign product_pipe = product_dsp;
        end else begin : lut_multiplier
            (* use_dsp = "no" *) logic signed [15:0] product_lut;
            always_ff @(posedge clk) begin
                if (!rst_n || clear) product_lut <= '0;
                else if (operand_valid) product_lut <= operand_a_pipe * operand_b_pipe;
            end
            assign product_pipe = product_lut;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            acc_out <= '0;
            product_valid <= 1'b0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            if (clear) begin
                acc_out <= '0;
                product_valid <= 1'b0;
            end else begin
                if (product_valid) begin
                    acc_out <= acc_out + {{16{product_pipe[15]}}, product_pipe};
                end
                product_valid <= operand_valid;
            end
        end
    end

endmodule
