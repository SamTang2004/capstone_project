module rowwise_norm_softmax_engine #(
    parameter int MAX_LENGTH = 1024,
    parameter int LENGTH_WIDTH = $clog2(MAX_LENGTH + 1)
) (
    input  logic clk, rst_n, start, softmax_mode,
    input  logic [LENGTH_WIDTH-1:0] vector_length,
    input  logic [31:0] epsilon_term,
    input  logic input_valid,
    output logic input_ready,
    input  logic signed [7:0] input_data,
    input  logic param_valid,
    output logic param_ready,
    input  logic signed [15:0] gamma_q8, beta_q8,
    output logic output_valid,
    input  logic output_ready,
    output logic signed [7:0] output_data,
    output logic output_last, busy, done, error
);
    typedef enum logic [4:0] {
        IDLE, LOAD, LN_STATS, LN_VAR_SUB, LN_VAR_EPS,
        LN_SQRT_TRIAL, LN_SQRT_COMPARE, LN_SQRT_SUB, LN_SQRT_UPDATE,
        SM_EXP_LOOKUP, SM_EXP_SUM, RECIP_PREP, RECIP_START, RECIP_WAIT,
        LN_CENTER, LN_NUMERATOR, SM_EXP_READ, SM_NUMERATOR,
        QUOTIENT_MUL, QUOTIENT_ESTIMATE, QUOTIENT_CHECK_MUL,
        QUOTIENT_REMAINDER, QUOTIENT_DECIDE, QUOTIENT_CORRECT,
        SM_OUTPUT, LN_SIGN, LN_AFFINE, LN_SCALE, LN_ROUND,
        EMIT
    } state_t;
    state_t state;

    logic mode_r;
    logic [LENGTH_WIDTH-1:0] length_r, index;
    logic signed [7:0] samples [0:MAX_LENGTH-1];
    logic [15:0] exponentials [0:255];
    logic signed [39:0] sum;
    logic [47:0] sum_squares;
    logic signed [15:0] input_square;
    logic signed [7:0] maximum;
    logic [31:0] exponential_sum;
    logic [15:0] current_exp, exponential_hold, exponential_read_hold;

    logic [63:0] sqrt_op, sqrt_res, sqrt_one, sqrt_trial;
    logic [63:0] variance_product_a, variance_product_b, variance_hold;
    logic sqrt_take, sqrt_low_borrow;
    logic [31:0] sqrt_low_difference;
    logic [5:0] sqrt_count;
    logic [31:0] denominator, reciprocal_divisor;

    logic reciprocal_start, reciprocal_busy, reciprocal_done, reciprocal_zero;
    logic [63:0] reciprocal_quotient, reciprocal_remainder;
    logic [32:0] reciprocal_q32;
    logic [32:0] numerator_abs;
    logic numerator_negative;
    logic [65:0] reciprocal_product;
    logic [32:0] quotient_estimate;
    logic [64:0] quotient_product;
    logic [32:0] quotient_remainder, corrected_hold;
    logic correction_needed;

    logic signed [39:0] sample_product;
    logic signed [63:0] centered;
    logic signed [15:0] gamma_hold, beta_hold;
    logic signed [33:0] signed_quotient_hold;
    logic signed [79:0] affine_product;
    logic signed [63:0] scaled_hold, rounded_value;
    logic signed [7:0] output_hold;
    logic [32:0] quotient_remainder_next;

    function automatic [15:0] exp_q15(input logic signed [8:0] delta);
        integer distance;
        begin
            distance = -delta;
            case (distance)
                0: exp_q15=32767; 1: exp_q15=25520; 2: exp_q15=19874;
                3: exp_q15=15480; 4: exp_q15=12055; 5: exp_q15=9388;
                6: exp_q15=7312; 7: exp_q15=5695; 8: exp_q15=4435;
                9: exp_q15=3454; 10: exp_q15=2690; 11: exp_q15=2095;
                12: exp_q15=1631; 13: exp_q15=1270; 14: exp_q15=989;
                15: exp_q15=770; 16: exp_q15=600; default: exp_q15=0;
            endcase
        end
    endfunction

    iterative_unsigned_divider #(.WIDTH(64)) reciprocal_divider (
        .clk, .rst_n, .start(reciprocal_start),
        .dividend(64'h0000_0001_0000_0000),
        .divisor({32'd0, reciprocal_divisor}),
        .busy(reciprocal_busy), .done(reciprocal_done),
        .divide_by_zero(reciprocal_zero),
        .quotient(reciprocal_quotient), .remainder(reciprocal_remainder)
    );

    assign input_square = input_data * input_data;
    assign current_exp = exp_q15($signed(samples[index]) - $signed(maximum));
    assign reciprocal_start = state == RECIP_START && !reciprocal_busy;
    assign input_ready = state == LOAD;
    assign param_ready = state == LN_CENTER && !mode_r;
    assign output_valid = state == EMIT;
    assign output_data = output_hold;
    assign output_last = output_valid && index == length_r - 1'b1;
    assign busy = state != IDLE;
    assign rounded_value = scaled_hold >= 0 ? scaled_hold + 128 : scaled_hold - 128;
    assign quotient_remainder_next = numerator_abs - quotient_product[32:0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; mode_r <= 1'b0; length_r <= '0; index <= '0;
            sum <= '0; sum_squares <= '0; maximum <= -8'sd128;
            exponential_sum <= '0; exponential_hold <= '0; exponential_read_hold <= '0;
            sqrt_op <= '0; sqrt_res <= '0;
            sqrt_one <= '0; sqrt_trial <= '0; variance_product_a <= '0;
            variance_product_b <= '0; variance_hold <= '0; sqrt_take <= 1'b0;
            sqrt_low_borrow <= 1'b0; sqrt_low_difference <= '0;
            sqrt_count <= '0; denominator <= '0; reciprocal_divisor <= '0;
            reciprocal_q32 <= '0; numerator_abs <= '0;
            numerator_negative <= 1'b0; reciprocal_product <= '0;
            quotient_estimate <= '0; quotient_product <= '0; quotient_remainder <= '0;
            correction_needed <= 1'b0; corrected_hold <= '0;
            sample_product <= '0; centered <= '0; gamma_hold <= '0;
            beta_hold <= '0; signed_quotient_hold <= '0;
            affine_product <= '0; scaled_hold <= '0;
            output_hold <= '0; done <= 1'b0; error <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (vector_length == 0 || vector_length > MAX_LENGTH ||
                        (softmax_mode && vector_length > 256)) begin
                        error <= 1'b1; done <= 1'b1;
                    end else begin
                        mode_r <= softmax_mode; length_r <= vector_length;
                        index <= 0; sum <= 0; sum_squares <= 0;
                        maximum <= -8'sd128; exponential_sum <= 0;
                        state <= LOAD;
                    end
                end
                LOAD: if (input_valid) begin
                    samples[index] <= input_data;
                    if (mode_r) begin
                        if (input_data > maximum) maximum <= input_data;
                    end else begin
                        sum <= sum + input_data;
                        sum_squares <= sum_squares + $unsigned(input_square);
                    end
                    if (index == length_r - 1'b1) begin
                        index <= 0;
                        state <= mode_r ? SM_EXP_LOOKUP : LN_STATS;
                    end else index <= index + 1'b1;
                end
                LN_STATS: begin
                    variance_product_a <= length_r * sum_squares;
                    variance_product_b <= sum * sum;
                    state <= LN_VAR_SUB;
                end
                LN_VAR_SUB: begin
                    variance_hold <= variance_product_a - variance_product_b;
                    state <= LN_VAR_EPS;
                end
                LN_VAR_EPS: begin
                    sqrt_op <= variance_hold + epsilon_term;
                    sqrt_res <= 0; sqrt_one <= 64'h4000_0000_0000_0000;
                    sqrt_count <= 0; state <= LN_SQRT_TRIAL;
                end
                LN_SQRT_TRIAL: begin
                    sqrt_trial <= sqrt_res + sqrt_one;
                    state <= LN_SQRT_COMPARE;
                end
                LN_SQRT_COMPARE: begin
                    sqrt_take <= sqrt_op >= sqrt_trial;
                    state <= LN_SQRT_SUB;
                end
                LN_SQRT_SUB: begin
                    sqrt_low_difference <= sqrt_op[31:0] - sqrt_trial[31:0];
                    sqrt_low_borrow <= sqrt_op[31:0] < sqrt_trial[31:0];
                    state <= LN_SQRT_UPDATE;
                end
                LN_SQRT_UPDATE: begin
                    if (sqrt_take) begin
                        sqrt_op <= {sqrt_op[63:32] - sqrt_trial[63:32] - sqrt_low_borrow,
                                    sqrt_low_difference};
                        sqrt_res <= (sqrt_res >> 1) + sqrt_one;
                        if (sqrt_count == 31)
                            denominator <= (sqrt_res >> 1) + sqrt_one;
                    end else begin
                        sqrt_res <= sqrt_res >> 1;
                        if (sqrt_count == 31) denominator <= sqrt_res >> 1;
                    end
                    sqrt_one <= sqrt_one >> 2;
                    if (sqrt_count == 31) begin index <= 0; state <= RECIP_PREP; end
                    else begin sqrt_count <= sqrt_count + 1'b1; state <= LN_SQRT_TRIAL; end
                end
                SM_EXP_LOOKUP: begin
                    // Prime the streaming lookup/accumulate pipeline.  Keeping
                    // the sample RAM and exponential LUT out of the 32-bit sum
                    // carry chain preserves one-element-per-cycle throughput
                    // after this single preload cycle.
                    exponentials[index] <= current_exp;
                    exponential_hold <= current_exp;
                    index <= index + 1'b1;
                    state <= SM_EXP_SUM;
                end
                SM_EXP_SUM: begin
                    exponential_sum <= exponential_sum + exponential_hold;
                    if (index == length_r) begin
                        index <= 0;
                        state <= RECIP_PREP;
                    end else begin
                        exponentials[index] <= current_exp;
                        exponential_hold <= current_exp;
                        index <= index + 1'b1;
                    end
                end
                RECIP_PREP: begin
                    reciprocal_divisor <= mode_r ? exponential_sum : denominator;
                    state <= RECIP_START;
                end
                RECIP_START: begin
                    if (!reciprocal_busy) state <= RECIP_WAIT;
                end
                RECIP_WAIT: if (reciprocal_done) begin
                    if (reciprocal_zero) begin error <= 1'b1; done <= 1'b1; state <= IDLE; end
                    else begin
                        reciprocal_q32 <= reciprocal_quotient[32:0];
                        index <= 0; state <= mode_r ? SM_EXP_READ : LN_CENTER;
                    end
                end
                LN_CENTER: if (param_valid) begin
                    gamma_hold <= gamma_q8; beta_hold <= beta_q8;
                    sample_product <= $signed(samples[index]) * $signed({1'b0, length_r});
                    state <= LN_NUMERATOR;
                end
                LN_NUMERATOR: begin
                    centered <= sample_product - sum;
                    numerator_negative <= sample_product - sum < 0;
                    numerator_abs <= sample_product - sum < 0
                        ? -((sample_product - sum) <<< 14)
                        : ((sample_product - sum) <<< 14);
                    state <= QUOTIENT_MUL;
                end
                SM_EXP_READ: begin
                    exponential_read_hold <= exponentials[index];
                    state <= SM_NUMERATOR;
                end
                SM_NUMERATOR: begin
                    numerator_abs <= exponential_read_hold * 127 + (exponential_sum >> 1);
                    state <= QUOTIENT_MUL;
                end
                QUOTIENT_MUL: begin
                    reciprocal_product <= numerator_abs * reciprocal_q32;
                    state <= QUOTIENT_ESTIMATE;
                end
                QUOTIENT_ESTIMATE: begin
                    quotient_estimate <= reciprocal_product >> 32;
                    state <= QUOTIENT_CHECK_MUL;
                end
                QUOTIENT_CHECK_MUL: begin
                    quotient_product <= quotient_estimate * reciprocal_divisor;
                    state <= QUOTIENT_REMAINDER;
                end
                QUOTIENT_REMAINDER: begin
                    quotient_remainder <= quotient_remainder_next;
                    correction_needed <= quotient_remainder_next >= reciprocal_divisor;
                    corrected_hold <= quotient_estimate +
                        (quotient_remainder_next >= reciprocal_divisor);
                    state <= mode_r ? SM_OUTPUT : LN_SIGN;
                end
                SM_OUTPUT: begin
                    output_hold <= corrected_hold > 127
                        ? 8'sd127 : $signed(corrected_hold[7:0]);
                    state <= EMIT;
                end
                LN_SIGN: begin
                    signed_quotient_hold <= numerator_negative
                        ? -$signed(corrected_hold)
                        : $signed(corrected_hold);
                    state <= LN_AFFINE;
                end
                LN_AFFINE: begin
                    affine_product <= signed_quotient_hold * gamma_hold;
                    state <= LN_SCALE;
                end
                LN_SCALE: begin
                    scaled_hold <= (affine_product >>> 14) + beta_hold;
                    state <= LN_ROUND;
                end
                LN_ROUND: begin
                    if ((rounded_value >>> 8) > 127) output_hold <= 8'sd127;
                    else if ((rounded_value >>> 8) < -128) output_hold <= -8'sd128;
                    else output_hold <= rounded_value[15:8];
                    state <= EMIT;
                end
                EMIT: if (output_ready) begin
                    if (index == length_r - 1'b1) begin state <= IDLE; done <= 1'b1; end
                    else begin
                        index <= index + 1'b1;
                        state <= mode_r ? SM_EXP_READ : LN_CENTER;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
