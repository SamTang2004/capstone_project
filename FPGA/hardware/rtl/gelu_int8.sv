module gelu_int8 #(
    parameter int THRESHOLD = 24
) (
    input  logic        input_valid,
    output logic        input_ready,
    input  logic [63:0] input_vector,
    output logic        output_valid,
    input  logic        output_ready,
    output logic [63:0] output_vector
);
    function automatic logic signed [7:0] gelu_lane(input logic signed [7:0] x);
        begin
            if (x <= -THRESHOLD) gelu_lane = 0;
            else if (x >= THRESHOLD) gelu_lane = x;
            // Exact lookup of the former quadratic approximation for Q4.4
            // values inside (-24, 24). This removes eight long combinational
            // multiply/divide paths from the packed datapath.
            else case (x)
                -8'sd23,-8'sd1,8'sd0: gelu_lane=0;
                -8'sd22,-8'sd21,-8'sd3,-8'sd2: gelu_lane=-8'sd1;
                -8'sd20,-8'sd19,-8'sd18,-8'sd17,-8'sd7,-8'sd6,-8'sd5,-8'sd4: gelu_lane=-8'sd2;
                -8'sd16,-8'sd15,-8'sd14,-8'sd13,-8'sd12,-8'sd11,-8'sd10,-8'sd9,-8'sd8: gelu_lane=-8'sd3;
                8'sd1,8'sd2: gelu_lane=8'sd1;
                8'sd3,8'sd4: gelu_lane=8'sd2;
                8'sd5: gelu_lane=8'sd3; 8'sd6: gelu_lane=8'sd4;
                8'sd7: gelu_lane=8'sd5; 8'sd8: gelu_lane=8'sd5;
                8'sd9: gelu_lane=8'sd6; 8'sd10: gelu_lane=8'sd7;
                8'sd11: gelu_lane=8'sd8; 8'sd12: gelu_lane=8'sd9;
                8'sd13: gelu_lane=8'sd10; 8'sd14: gelu_lane=8'sd11;
                8'sd15: gelu_lane=8'sd12; 8'sd16: gelu_lane=8'sd13;
                8'sd17: gelu_lane=8'sd15; 8'sd18: gelu_lane=8'sd16;
                8'sd19: gelu_lane=8'sd17; 8'sd20: gelu_lane=8'sd18;
                8'sd21: gelu_lane=8'sd20; 8'sd22: gelu_lane=8'sd21;
                default: gelu_lane=8'sd23;
            endcase
        end
    endfunction

    assign input_ready = output_ready;
    assign output_valid = input_valid;
    always_comb begin
        for (int lane = 0; lane < 8; lane++)
            output_vector[lane*8 +: 8] = gelu_lane($signed(input_vector[lane*8 +: 8]));
    end
endmodule
