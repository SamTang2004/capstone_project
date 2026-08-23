module weight_vector_decode (
    input  logic        weight_int4,
    input  logic [63:0] packed_weights,
    output logic [63:0] decoded_weights
);
    always_comb begin
        decoded_weights = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (weight_int4) begin
                decoded_weights[lane*8 +: 8] = {
                    {4{packed_weights[lane*4 + 3]}},
                    packed_weights[lane*4 +: 4]
                };
            end else begin
                decoded_weights[lane*8 +: 8] = packed_weights[lane*8 +: 8];
            end
        end
    end
endmodule
