module pe_int8_mac (
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

    // A 16x16 array has 256 PEs, exceeding the PYNQ-Z2's 220 DSP48s.
    // Keep this registered multiplier in LUT fabric until the physical array
    // is resized or time-multiplexed for a DSP-budgeted implementation.
    (* use_dsp = "no" *) logic signed [15:0] product_pipe;
    logic product_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            acc_out <= '0;
            product_pipe <= '0;
            product_valid <= 1'b0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            if (clear) begin
                acc_out <= '0;
                product_pipe <= '0;
                product_valid <= 1'b0;
            end else begin
                if (product_valid) begin
                    acc_out <= acc_out + {{16{product_pipe[15]}}, product_pipe};
                end
                if (enable) begin
                    product_pipe <= a_in * b_in;
                    product_valid <= 1'b1;
                end else begin
                    product_valid <= 1'b0;
                end
            end
        end
    end

endmodule
