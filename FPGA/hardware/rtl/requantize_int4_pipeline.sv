`timescale 1ns/1ps
module requantize_int4_pipeline (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    input  logic signed [31:0] input_value,
    input  logic signed [31:0] multiplier,
    input  logic [5:0]         right_shift,
    input  logic signed [31:0] output_zero_point,
    output logic               output_valid,
    output logic signed [3:0]  output_value
);
    logic valid_multiply;
    logic valid_scale;
    logic valid_offset;
    (* use_dsp = "yes" *) logic signed [63:0] product_r;
    logic signed [63:0] scaled_r;
    logic signed [63:0] offset_r;
    logic [5:0] shift_r;
    logic signed [31:0] zero_point_r;

    function automatic logic signed [3:0] saturate_int4(input logic signed [63:0] value);
        if (value < -64'sd8)
            saturate_int4 = -4'sd8;
        else if (value > 64'sd7)
            saturate_int4 = 4'sd7;
        else
            saturate_int4 = value[3:0];
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_multiply <= 1'b0;
            valid_scale <= 1'b0;
            valid_offset <= 1'b0;
            output_valid <= 1'b0;
            product_r <= '0;
            scaled_r <= '0;
            offset_r <= '0;
            shift_r <= '0;
            zero_point_r <= '0;
            output_value <= '0;
        end else begin
            valid_multiply <= input_valid;
            valid_scale <= valid_multiply;
            valid_offset <= valid_scale;
            output_valid <= valid_offset;

            if (input_valid) begin
                product_r <= input_value * multiplier;
                shift_r <= right_shift;
                zero_point_r <= output_zero_point;
            end
            if (valid_multiply)
                scaled_r <= product_r >>> shift_r;
            if (valid_scale)
                offset_r <= scaled_r + zero_point_r;
            if (valid_offset)
                output_value <= saturate_int4(offset_r);
        end
    end
endmodule
