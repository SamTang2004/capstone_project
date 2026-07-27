`timescale 1ns/1ps
module int4_output_postprocess (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    output logic               input_ready,
    input  logic signed [31:0] input_data,
    input  logic               input_array,
    input  logic [5:0]         input_addr,
    input  logic signed [31:0] multiplier,
    input  logic [5:0]         right_shift,
    input  logic signed [31:0] output_zero_point,
    input  logic               activation_enable,
    input  logic               leaky_mode,
    input  logic [2:0]         leaky_shift,
    output logic               output_valid,
    input  logic               output_ready,
    output logic signed [3:0]  output_data,
    output logic               output_array,
    output logic [5:0]         output_addr
);

    logic pipeline_valid;
    logic signed [3:0] pipeline_data;
    logic metadata_array;
    logic [5:0] metadata_addr;
    logic holding_valid;
    logic in_flight;
    logic signed [3:0] activated_data;

    assign input_ready = !in_flight;
    assign output_valid = holding_valid;

    always_comb begin
        activated_data = pipeline_data;
        if (activation_enable && pipeline_data < 0)
            activated_data = leaky_mode ? pipeline_data >>> leaky_shift : 4'sd0;
    end

    requantize_int4_pipeline requantizer (
        .clk, .rst_n,
        .input_valid(input_valid && input_ready),
        .input_value(input_data),
        .multiplier,
        .right_shift,
        .output_zero_point,
        .output_valid(pipeline_valid),
        .output_value(pipeline_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            metadata_array <= 1'b0;
            metadata_addr <= '0;
            holding_valid <= 1'b0;
            in_flight <= 1'b0;
            output_data <= '0;
            output_array <= 1'b0;
            output_addr <= '0;
        end else begin
            if (input_valid && input_ready) begin
                in_flight <= 1'b1;
                metadata_array <= input_array;
                metadata_addr <= input_addr;
            end

            if (pipeline_valid) begin
                holding_valid <= 1'b1;
                output_data <= activated_data;
                output_array <= metadata_array;
                output_addr <= metadata_addr;
            end else if (holding_valid && output_ready) begin
                holding_valid <= 1'b0;
                in_flight <= 1'b0;
            end
        end
    end
endmodule
