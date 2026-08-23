module gemm_8x8_tile_core (
    input  logic          clk,
    input  logic          rst_n,

    input  logic          start,
    input  logic          weight_int4,
    input  logic [15:0]   k_steps,
    input  logic [3:0]    valid_rows,
    input  logic [3:0]    valid_cols,

    input  logic          input_valid,
    output logic          input_ready,
    input  logic [63:0]   activation_vector,
    input  logic [63:0]   packed_weight_vector,

    output logic          busy,
    output logic          result_valid,
    input  logic          result_ready,
    output logic          error,
    output logic [2047:0] accumulators
);
    typedef enum logic [2:0] {IDLE, CLEAR, RUN, DRAIN, RESULT, FAIL} state_t;
    state_t state;

    logic weight_int4_r;
    logic [15:0] k_steps_r, accepted_steps;
    logic [3:0] valid_rows_r, valid_cols_r;
    logic drain_cycle;
    logic [63:0] decoded_weights, masked_activations, masked_weights;
    logic array_clear, array_step;

    weight_vector_decode decode (
        .weight_int4(weight_int4_r),
        .packed_weights(packed_weight_vector),
        .decoded_weights(decoded_weights)
    );

    always_comb begin
        masked_activations = '0;
        masked_weights = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (lane < valid_rows_r)
                masked_activations[lane*8 +: 8] = activation_vector[lane*8 +: 8];
            if (lane < valid_cols_r)
                masked_weights[lane*8 +: 8] = decoded_weights[lane*8 +: 8];
        end
    end

    assign input_ready = state == RUN;
    assign array_clear = state == CLEAR;
    assign array_step = input_valid && input_ready;
    assign busy = state != IDLE && state != RESULT && state != FAIL;
    assign result_valid = state == RESULT;

    matrix_array_8x8 array (
        .clk(clk),
        .rst_n(rst_n),
        .clear(array_clear),
        .step_valid(array_step),
        .activation_vector(masked_activations),
        .weight_vector(masked_weights),
        .accumulators(accumulators)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            weight_int4_r <= 1'b0;
            k_steps_r <= '0;
            accepted_steps <= '0;
            valid_rows_r <= '0;
            valid_cols_r <= '0;
            drain_cycle <= 1'b0;
            error <= 1'b0;
        end else begin
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        weight_int4_r <= weight_int4;
                        k_steps_r <= k_steps;
                        valid_rows_r <= valid_rows;
                        valid_cols_r <= valid_cols;
                        accepted_steps <= '0;
                        drain_cycle <= 1'b0;
                        if (k_steps == 0 || valid_rows == 0 || valid_rows > 8 ||
                            valid_cols == 0 || valid_cols > 8) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else begin
                            state <= CLEAR;
                        end
                    end
                end
                CLEAR: state <= RUN;
                RUN: if (array_step) begin
                    accepted_steps <= accepted_steps + 1'b1;
                    if (accepted_steps + 1'b1 == k_steps_r) begin
                        drain_cycle <= 1'b0;
                        state <= DRAIN;
                    end
                end
                DRAIN: begin
                    if (drain_cycle) state <= RESULT;
                    else drain_cycle <= 1'b1;
                end
                RESULT: if (result_ready) state <= IDLE;
                FAIL: if (!start) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
