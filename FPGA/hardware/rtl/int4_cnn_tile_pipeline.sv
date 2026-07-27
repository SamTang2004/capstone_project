`timescale 1ns/1ps
module int4_cnn_tile_pipeline (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               load_valid,
    input  logic               load_array,
    input  logic               load_matrix,
    input  logic [5:0]         load_addr,
    input  logic signed [3:0]  load_data,
    input  logic               bias_load_valid,
    input  logic               bias_load_array,
    input  logic [2:0]         bias_load_col,
    input  logic signed [31:0] bias_load_data,
    input  logic               start,
    input  logic [1:0]         share_mode,
    input  logic signed [31:0] multiplier,
    input  logic [5:0]         right_shift,
    input  logic signed [31:0] output_zero_point,
    input  logic               leaky_mode,
    input  logic [2:0]         leaky_shift,
    output logic               busy,
    output logic               done,
    output logic               output_valid,
    output logic               output_array,
    output logic [5:0]         output_addr,
    output logic signed [3:0]  output_data
);
    typedef enum logic [1:0] {IDLE, COMPUTE, DRAIN, FINISH} state_t;
    state_t state;
    logic core_start;
    logic core_busy;
    logic core_done;
    logic core_result_array;
    logic [5:0] core_result_addr;
    logic signed [31:0] core_result_data;
    logic [7:0] drain_index;
    logic signed [31:0] bias [0:1][0:7];
    logic signed [31:0] multiplier_r;
    logic [5:0] right_shift_r;
    logic signed [31:0] output_zero_point_r;
    logic leaky_mode_r;
    logic [2:0] leaky_shift_r;
    logic requant_input_valid;
    logic signed [31:0] requant_input;
    logic requant_output_valid;
    logic signed [3:0] requant_output;
    logic [3:0] metadata_valid;
    logic [6:0] metadata [0:3];

    assign busy = state != IDLE;
    assign core_result_array = drain_index[6];
    assign core_result_addr = drain_index[5:0];
    assign requant_input_valid = state == DRAIN && drain_index < 8'd128;
    assign requant_input = core_result_data + bias[core_result_array][core_result_addr[2:0]];

    gemm_dual_8x8_int4_core core (
        .clk(clk), .rst_n(rst_n),
        .load_valid(load_valid), .load_array(load_array), .load_matrix(load_matrix),
        .load_addr(load_addr), .load_data(load_data),
        .load_vector_valid(1'b0), .load_vector_ready(),
        .load_vector_array(1'b0), .load_vector_matrix(1'b0),
        .load_vector_row(3'd0), .load_vector_mask(8'd0),
        .load_vector_data(32'd0),
        .load_group_valid(1'b0), .load_group_ready(),
        .load_group_row(3'd0),
        .load_group_a_mask(8'd0), .load_group_a_data(32'd0),
        .load_group_b0_mask(8'd0), .load_group_b0_data(32'd0),
        .load_group_b1_mask(8'd0), .load_group_b1_data(32'd0),
        .start(core_start), .share_mode(share_mode),
        .busy(core_busy), .done(core_done),
        .result_array(core_result_array), .result_addr(core_result_addr),
        .result_data(core_result_data),
        .result_row(3'd0), .result0_row(), .result1_row()
    );

    requantize_int4_pipeline requantizer (
        .clk(clk), .rst_n(rst_n),
        .input_valid(requant_input_valid), .input_value(requant_input),
        .multiplier(multiplier_r), .right_shift(right_shift_r),
        .output_zero_point(output_zero_point_r),
        .output_valid(requant_output_valid), .output_value(requant_output)
    );

    always_comb begin
        output_valid = requant_output_valid && metadata_valid[3];
        output_array = metadata[3][6];
        output_addr = metadata[3][5:0];
        if (requant_output < 0)
            output_data = leaky_mode_r ? requant_output >>> leaky_shift_r : 4'sd0;
        else
            output_data = requant_output;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            core_start <= 1'b0;
            drain_index <= '0;
            multiplier_r <= '0;
            right_shift_r <= '0;
            output_zero_point_r <= '0;
            leaky_mode_r <= 1'b0;
            leaky_shift_r <= '0;
            metadata_valid <= '0;
            for (int stage = 0; stage < 4; stage++) metadata[stage] <= '0;
            for (int array_idx = 0; array_idx < 2; array_idx++)
                for (int col = 0; col < 8; col++) bias[array_idx][col] <= '0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            core_start <= 1'b0;
            metadata_valid[0] <= requant_input_valid;
            metadata[0] <= drain_index[6:0];
            for (int stage = 1; stage < 4; stage++) begin
                metadata_valid[stage] <= metadata_valid[stage-1];
                metadata[stage] <= metadata[stage-1];
            end

            if (bias_load_valid && state == IDLE)
                bias[bias_load_array][bias_load_col] <= bias_load_data;

            unique case (state)
                IDLE: if (start) begin
                    multiplier_r <= multiplier;
                    right_shift_r <= right_shift;
                    output_zero_point_r <= output_zero_point;
                    leaky_mode_r <= leaky_mode;
                    leaky_shift_r <= leaky_shift;
                    core_start <= 1'b1;
                    state <= COMPUTE;
                end
                COMPUTE: if (core_done) begin
                    drain_index <= '0;
                    state <= DRAIN;
                end
                DRAIN: begin
                    if (requant_input_valid)
                        drain_index <= drain_index + 1'b1;
                    if (output_valid && output_array && output_addr == 6'd63)
                        state <= FINISH;
                end
                FINISH: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
