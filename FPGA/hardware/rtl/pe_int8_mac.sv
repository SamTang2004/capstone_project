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

    logic signed [15:0] product_pipe;
    logic product_valid;

    generate
        if (USE_DSP) begin : dsp_multiplier
            (* use_dsp = "yes" *) logic signed [15:0] product_dsp;
            always_ff @(posedge clk) begin
                if (!rst_n || clear) product_dsp <= '0;
                else if (enable) product_dsp <= a_in * b_in;
            end
            assign product_pipe = product_dsp;
        end else begin : lut_multiplier
            (* use_dsp = "no" *) logic signed [15:0] product_lut;
            always_ff @(posedge clk) begin
                if (!rst_n || clear) product_lut <= '0;
                else if (enable) product_lut <= a_in * b_in;
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
                if (enable) begin
                    product_valid <= 1'b1;
                end else begin
                    product_valid <= 1'b0;
                end
            end
        end
    end

endmodule
