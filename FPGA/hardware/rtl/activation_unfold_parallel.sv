module activation_unfold_parallel #(
    parameter int LANES = 16
) (
    input  logic                         enable,
    input  logic                         leaky_mode,
    input  logic [2:0]                   leaky_shift,
    input  logic signed [LANES*8-1:0]    input_flat,
    output logic signed [LANES*8-1:0]    output_flat
);
    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [7:0] value;
            value = input_flat[lane*8 +: 8];
            if (!enable || value >= 0) begin
                output_flat[lane*8 +: 8] = value;
            end else if (leaky_mode) begin
                output_flat[lane*8 +: 8] = value >>> leaky_shift;
            end else begin
                output_flat[lane*8 +: 8] = '0;
            end
        end
    end
endmodule
