module rowwise_norm_softmax_engine #(
    parameter int MAX_LENGTH = 1024,
    parameter int LENGTH_WIDTH = $clog2(MAX_LENGTH + 1)
) (
    input logic clk, rst_n, start, softmax_mode,
    input logic [LENGTH_WIDTH-1:0] vector_length,
    input logic [31:0] epsilon_term,
    input logic input_valid,
    output logic input_ready,
    input logic signed [7:0] input_data,
    input logic param_valid,
    output logic param_ready,
    input logic signed [15:0] gamma_q8, beta_q8,
    output logic output_valid,
    input logic output_ready,
    output logic signed [7:0] output_data,
    output logic output_last, busy, done, error
);
    typedef enum logic [3:0] {
        IDLE, LOAD, LN_STATS, LN_VAR_SUB, LN_VAR_EPS,
        LN_SQRT_TRIAL, LN_SQRT_COMPARE, LN_SQRT_SUB, LN_SQRT_UPDATE,
        SM_EXP_LOOKUP, SM_EXP_SUM, RECIP_PREP, RECIP_START, RECIP_WAIT,
        PROCESS
    } state_t;
    state_t state;

    logic mode_r;
    logic [LENGTH_WIDTH-1:0] length_r, index, feed_index;
    logic feed_done;
    logic signed [7:0] samples [0:MAX_LENGTH-1];
    logic [15:0] exponentials [0:255];
    logic signed [39:0] sum;
    logic [47:0] sum_squares;
    logic signed [15:0] input_square;
    logic signed [7:0] maximum;
    logic [31:0] exponential_sum;
    logic [15:0] current_exp, exponential_hold;

    logic [63:0] sqrt_op, sqrt_res, sqrt_one, sqrt_trial;
    logic [63:0] variance_product_a, variance_product_b, variance_hold;
    logic sqrt_take, sqrt_low_borrow;
    logic [31:0] sqrt_low_difference;
    logic [5:0] sqrt_count;
    logic [31:0] denominator, reciprocal_divisor;
    logic reciprocal_start, reciprocal_busy, reciprocal_done, reciprocal_zero;
    logic [63:0] reciprocal_quotient, reciprocal_remainder;
    logic [32:0] reciprocal_q32;

    // Exact quotient and affine pipeline. The former engine serialized these
    // stages for every element; this version accepts one new element per cycle.
    logic [6:0] pipe_valid, pipe_last;
    logic signed [7:0] pf_sample;
    logic [15:0] pf_exponential;
    logic signed [15:0] pf_gamma, pf_beta;
    logic [32:0] p0_numerator, p1_numerator, p2_numerator;
    logic p0_negative, p1_negative, p2_negative;
    logic signed [15:0] p0_gamma, p1_gamma, p2_gamma, p3_gamma;
    logic signed [15:0] p0_beta, p1_beta, p2_beta, p3_beta, p4_beta;
    logic p3_negative;
    logic [65:0] p1_reciprocal_product;
    logic [32:0] p2_estimate, p3_corrected;
    logic [64:0] p2_quotient_product;
    logic signed [79:0] p4_affine;
    logic signed [7:0] p4_output, p5_output;
    logic pipe_advance, feed_valid;
    logic [32:0] corrected_wire;

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

    function automatic logic [32:0] absolute_ln_numerator(
        input logic signed [7:0] sample,
        input logic [LENGTH_WIDTH-1:0] length,
        input logic signed [39:0] row_sum
    );
        logic signed [39:0] centered_value;
        logic signed [53:0] shifted_value;
        begin
            centered_value = sample * $signed({1'b0, length}) - row_sum;
            shifted_value = centered_value <<< 14;
            absolute_ln_numerator = shifted_value < 0 ? -shifted_value : shifted_value;
        end
    endfunction

    function automatic logic signed [33:0] signed_quotient(
        input logic [32:0] value, input logic negative
    );
        begin
            signed_quotient = negative ? -$signed({1'b0, value}) : $signed({1'b0, value});
        end
    endfunction

    function automatic logic signed [7:0] finish_layernorm(
        input logic signed [79:0] affine,
        input logic signed [15:0] beta
    );
        logic signed [79:0] scaled, rounded, quantized;
        begin
            scaled = (affine >>> 14) + beta;
            rounded = scaled >= 0 ? scaled + 128 : scaled - 128;
            quantized = rounded >>> 8;
            if (quantized > 127) finish_layernorm = 8'sd127;
            else if (quantized < -128) finish_layernorm = -8'sd128;
            else finish_layernorm = quantized[7:0];
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
    assign pipe_advance = !pipe_valid[6] || output_ready;
    assign feed_valid = state == PROCESS && !feed_done && pipe_advance &&
        (mode_r || param_valid);
    assign param_ready = state == PROCESS && !mode_r && !feed_done && pipe_advance;
    assign output_valid = state == PROCESS && pipe_valid[6];
    assign output_data = p5_output;
    assign output_last = output_valid && pipe_last[6];
    assign busy = state != IDLE;
    assign corrected_wire = p2_estimate +
        ((p2_numerator - p2_quotient_product[32:0]) >= reciprocal_divisor);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; mode_r <= 0; length_r <= 0; index <= 0;
            feed_index <= 0; feed_done <= 0; sum <= 0; sum_squares <= 0;
            maximum <= -8'sd128; exponential_sum <= 0; exponential_hold <= 0;
            sqrt_op <= 0; sqrt_res <= 0; sqrt_one <= 0; sqrt_trial <= 0;
            variance_product_a <= 0; variance_product_b <= 0; variance_hold <= 0;
            sqrt_take <= 0; sqrt_low_borrow <= 0; sqrt_low_difference <= 0;
            sqrt_count <= 0; denominator <= 0; reciprocal_divisor <= 0;
            reciprocal_q32 <= 0; pipe_valid <= 0; pipe_last <= 0;
            pf_sample <= 0; pf_exponential <= 0; pf_gamma <= 0; pf_beta <= 0;
            p0_numerator <= 0; p1_numerator <= 0; p2_numerator <= 0;
            p0_negative <= 0; p1_negative <= 0; p2_negative <= 0;
            p0_gamma <= 0; p1_gamma <= 0; p2_gamma <= 0; p3_gamma <= 0;
            p0_beta <= 0; p1_beta <= 0; p2_beta <= 0; p3_beta <= 0; p4_beta <= 0;
            p3_negative <= 0;
            p1_reciprocal_product <= 0; p2_estimate <= 0;
            p2_quotient_product <= 0; p3_corrected <= 0;
            p4_affine <= 0; p4_output <= 0; p5_output <= 0; done <= 0; error <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: if (start) begin
                    error <= 0;
                    if (vector_length == 0 || vector_length > MAX_LENGTH ||
                        (softmax_mode && vector_length > 256)) begin
                        error <= 1; done <= 1;
                    end else begin
                        mode_r <= softmax_mode; length_r <= vector_length;
                        index <= 0; sum <= 0; sum_squares <= 0;
                        maximum <= -8'sd128; exponential_sum <= 0;
                        pipe_valid <= 0; feed_done <= 0; state <= LOAD;
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
                        index <= 0; state <= mode_r ? SM_EXP_LOOKUP : LN_STATS;
                    end else index <= index + 1'b1;
                end
                LN_STATS: begin
                    variance_product_a <= length_r * sum_squares;
                    variance_product_b <= sum * sum; state <= LN_VAR_SUB;
                end
                LN_VAR_SUB: begin variance_hold <= variance_product_a - variance_product_b; state <= LN_VAR_EPS; end
                LN_VAR_EPS: begin
                    sqrt_op <= variance_hold + epsilon_term;
                    sqrt_res <= 0; sqrt_one <= 64'h4000_0000_0000_0000;
                    sqrt_count <= 0; state <= LN_SQRT_TRIAL;
                end
                LN_SQRT_TRIAL: begin sqrt_trial <= sqrt_res + sqrt_one; state <= LN_SQRT_COMPARE; end
                LN_SQRT_COMPARE: begin sqrt_take <= sqrt_op >= sqrt_trial; state <= LN_SQRT_SUB; end
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
                        if (sqrt_count == 31) denominator <= (sqrt_res >> 1) + sqrt_one;
                    end else begin
                        sqrt_res <= sqrt_res >> 1;
                        if (sqrt_count == 31) denominator <= sqrt_res >> 1;
                    end
                    sqrt_one <= sqrt_one >> 2;
                    if (sqrt_count == 31) begin index <= 0; state <= RECIP_PREP; end
                    else begin sqrt_count <= sqrt_count + 1'b1; state <= LN_SQRT_TRIAL; end
                end
                SM_EXP_LOOKUP: begin
                    exponentials[index] <= current_exp; exponential_hold <= current_exp;
                    index <= index + 1'b1; state <= SM_EXP_SUM;
                end
                SM_EXP_SUM: begin
                    exponential_sum <= exponential_sum + exponential_hold;
                    if (index == length_r) begin index <= 0; state <= RECIP_PREP; end
                    else begin
                        exponentials[index] <= current_exp; exponential_hold <= current_exp;
                        index <= index + 1'b1;
                    end
                end
                RECIP_PREP: begin
                    reciprocal_divisor <= mode_r ? exponential_sum : denominator;
                    state <= RECIP_START;
                end
                RECIP_START: if (!reciprocal_busy) state <= RECIP_WAIT;
                RECIP_WAIT: if (reciprocal_done) begin
                    if (reciprocal_zero) begin error <= 1; done <= 1; state <= IDLE; end
                    else begin
                        reciprocal_q32 <= reciprocal_quotient[32:0];
                        feed_index <= 0; feed_done <= 0; pipe_valid <= 0;
                        state <= PROCESS;
                    end
                end
                PROCESS: begin
                    if (output_valid && output_ready && output_last) begin
                        pipe_valid <= 0; done <= 1; state <= IDLE;
                    end else if (pipe_advance) begin
                        pipe_valid <= {pipe_valid[5:0], feed_valid};
                        pipe_last <= {pipe_last[5:0], feed_index == length_r - 1'b1};

                        if (mode_r) begin
                            p0_numerator <= pf_exponential * 127 +
                                (exponential_sum >> 1);
                            p0_negative <= 0; p0_gamma <= 0; p0_beta <= 0;
                        end else begin
                            p0_numerator <= absolute_ln_numerator(pf_sample, length_r, sum);
                            p0_negative <= pf_sample *
                                $signed({1'b0, length_r}) - sum < 0;
                            p0_gamma <= pf_gamma; p0_beta <= pf_beta;
                        end

                        p1_numerator <= p0_numerator;
                        p1_negative <= p0_negative; p1_gamma <= p0_gamma; p1_beta <= p0_beta;
                        p1_reciprocal_product <= p0_numerator * reciprocal_q32;

                        p2_numerator <= p1_numerator;
                        p2_negative <= p1_negative; p2_gamma <= p1_gamma; p2_beta <= p1_beta;
                        p2_estimate <= p1_reciprocal_product >> 32;
                        p2_quotient_product <= (p1_reciprocal_product >> 32) * reciprocal_divisor;

                        p3_corrected <= corrected_wire; p3_negative <= p2_negative;
                        p3_gamma <= p2_gamma; p3_beta <= p2_beta;

                        if (mode_r)
                            p4_output <= p3_corrected > 127 ? 8'sd127 : p3_corrected[7:0];
                        else begin
                            p4_affine <= signed_quotient(p3_corrected, p3_negative) * p3_gamma;
                            p4_beta <= p3_beta;
                        end

                        if (mode_r)
                            p5_output <= p4_output;
                        else
                            p5_output <= finish_layernorm(p4_affine, p4_beta);

                        if (feed_valid) begin
                            if (mode_r) begin
                                pf_exponential <= exponentials[feed_index];
                                pf_sample <= 0; pf_gamma <= 0; pf_beta <= 0;
                            end else begin
                                pf_sample <= samples[feed_index];
                                pf_exponential <= 0;
                                pf_gamma <= gamma_q8; pf_beta <= beta_q8;
                            end
                            if (feed_index == length_r - 1'b1) feed_done <= 1;
                            else feed_index <= feed_index + 1'b1;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
