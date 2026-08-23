module bias_add_int32 (
    input  logic          input_valid,
    output logic          input_ready,
    input  logic [255:0]  accumulators,
    input  logic [255:0]  bias,
    output logic          output_valid,
    input  logic          output_ready,
    output logic [255:0]  result
);
    assign input_ready = output_ready;
    assign output_valid = input_valid;
    always_comb begin
        for (int lane = 0; lane < 8; lane++)
            result[lane*32 +: 32] = $signed(accumulators[lane*32 +: 32]) +
                                            $signed(bias[lane*32 +: 32]);
    end
endmodule
