module residual_add_int8 (
    input  logic        input_valid,
    output logic        input_ready,
    input  logic [63:0] lhs,
    input  logic [63:0] rhs,
    input  logic [3:0]  lhs_right_shift,
    input  logic [3:0]  rhs_right_shift,
    output logic        output_valid,
    input  logic        output_ready,
    output logic [63:0] result
);
    function automatic logic signed [7:0] saturate_int8(input logic signed [9:0] value);
        if (value > 127) saturate_int8 = 8'sd127;
        else if (value < -128) saturate_int8 = -8'sd128;
        else saturate_int8 = value[7:0];
    endfunction

    assign input_ready = output_ready;
    assign output_valid = input_valid;
    always_comb begin
        result = '0;
        for (int lane = 0; lane < 8; lane++) begin
            result[lane*8 +: 8] = saturate_int8(
                ($signed(lhs[lane*8 +: 8]) >>> lhs_right_shift) +
                ($signed(rhs[lane*8 +: 8]) >>> rhs_right_shift)
            );
        end
    end
endmodule
