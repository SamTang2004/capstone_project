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
    logic [7:0] active_rows_r, active_cols_r;
    logic drain_cycle;
    logic [63:0] decoded_weights, masked_activations, masked_weights;
    logic array_clear, array_step;

    // Decode the tile dimensions once at command acceptance.  Keeping the
    // binary counts live during RUN made each count bit feed comparisons for
    // all 64 PEs; the resulting 162-fanout route was the 100 MHz critical
    // path.  Each thermometer bit below controls only one array lane.
    function automatic logic [7:0] lane_mask(input logic [3:0] count);
        unique case (count)
            4'd1: lane_mask = 8'h01;
            4'd2: lane_mask = 8'h03;
            4'd3: lane_mask = 8'h07;
            4'd4: lane_mask = 8'h0f;
            4'd5: lane_mask = 8'h1f;
            4'd6: lane_mask = 8'h3f;
            4'd7: lane_mask = 8'h7f;
            4'd8: lane_mask = 8'hff;
            default: lane_mask = 8'h00;
        endcase
    endfunction

    weight_vector_decode decode (
        .weight_int4(weight_int4_r),
        .packed_weights(packed_weight_vector),
        .decoded_weights(decoded_weights)
    );

    always_comb begin
        masked_activations = '0;
        masked_weights = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (active_rows_r[lane])
                masked_activations[lane*8 +: 8] = activation_vector[lane*8 +: 8];
            if (active_cols_r[lane])
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
            active_rows_r <= '0;
            active_cols_r <= '0;
            drain_cycle <= 1'b0;
            error <= 1'b0;
        end else begin
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        weight_int4_r <= weight_int4;
                        k_steps_r <= k_steps;
                        active_rows_r <= lane_mask(valid_rows);
                        active_cols_r <= lane_mask(valid_cols);
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
