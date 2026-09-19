// Eight independent multipliers, eight accumulator contexts. Every beat may
// contribute multiple products to the same context; a balanced tree reduces them.
// Results are INT32 modulo 2^32, with no intermediate requantization.
module llm_dot_core8 #(parameter bit FIXED_LANE_SLOTS = 0) (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [7:0] cmd_job_mask,
    input logic [255:0] cmd_job_tags,
    input logic [15:0] cmd_wave_tag,
    input logic in_valid, output logic in_ready,
    input logic [7:0] in_mask,
    input logic [63:0] in_a, in_b,
    input logic [23:0] in_slots,
    input logic in_last,
    output logic result_valid, input logic result_ready,
    output logic [7:0] result_mask,
    output logic [255:0] result_tags, result_values,
    output logic [15:0] result_wave_tag
);
    // Must match AREG=2, BREG=2, MREG=1, PREG=1 in the DSP lane.
    localparam integer PRODUCT_STAGES = 4;
    logic active, last_seen, sum_valid, sum_last;
    logic [PRODUCT_STAGES-1:0] valid_pipe, last_pipe;
    logic [7:0] mask_pipe [0:PRODUCT_STAGES-1];
    logic [23:0] slots_pipe [0:PRODUCT_STAGES-1];
    wire pipe_valid = valid_pipe[PRODUCT_STAGES-1];
    wire pipe_last = last_pipe[PRODUCT_STAGES-1];
    wire [7:0] mask_q = mask_pipe[PRODUCT_STAGES-1];
    wire [23:0] slots_q = slots_pipe[PRODUCT_STAGES-1];
    logic [7:0] jobs_q;
    logic [255:0] tags_q;
    logic [15:0] wave_q;
    wire signed [15:0] products [0:7];
    logic [31:0] accum [0:7], accum_next [0:7];
    logic signed [18:0] sums_q [0:7];
    wire accept = in_valid && in_ready;
    assign cmd_ready = rst_n && !active && !result_valid;
    assign in_ready = rst_n && active && !last_seen;

    for (genvar lane = 0; lane < 8; lane++) begin : multipliers
        llm_dot_product_lane mul (.clk, .rst_n, .enable(accept && in_mask[lane]),
            .a(in_a[lane*8 +: 8]), .b(in_b[lane*8 +: 8]), .product(products[lane]));
    end
    for (genvar job = 0; job < 8; job++) begin : reducers
        wire signed [15:0] terms [0:7];
        wire signed [16:0] pairs [0:3];
        wire signed [17:0] quads [0:1];
        wire signed [18:0] total;
        for (genvar i = 0; i < 8; i++) begin
            // Activation broadcast assigns one output row to each lane. Resolve
            // that mapping at elaboration instead of routing an 8x8 reduction.
            assign terms[i] = (mask_q[i] && (FIXED_LANE_SLOTS ? i==job : slots_q[i*3 +: 3] == job)) ? products[i] : 16'sd0;
        end
        for (genvar i = 0; i < 4; i++) begin
            assign pairs[i] = {terms[2*i][15], terms[2*i]} + {terms[2*i+1][15], terms[2*i+1]};
        end
        for (genvar i = 0; i < 2; i++) begin
            assign quads[i] = {pairs[2*i][16], pairs[2*i]} + {pairs[2*i+1][16], pairs[2*i+1]};
        end
        assign total = {quads[0][17], quads[0]} + {quads[1][17], quads[1]};
        always_ff @(posedge clk) begin
            if (!rst_n) sums_q[job] <= 0;
            else if (pipe_valid) sums_q[job] <= total;
        end
        assign accum_next[job] = accum[job] + {{13{sums_q[job][18]}}, sums_q[job]};
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active <= 0; last_seen <= 0; valid_pipe <= 0; last_pipe <= 0;
            sum_valid <= 0; sum_last <= 0;
            jobs_q <= 0; tags_q <= 0; wave_q <= 0;
            for (integer s = 0; s < PRODUCT_STAGES; s++) begin
                mask_pipe[s] <= 0; slots_pipe[s] <= 0;
            end
            result_valid <= 0; result_mask <= 0; result_tags <= 0;
            result_values <= 0; result_wave_tag <= 0;
            for (integer j = 0; j < 8; j++) accum[j] <= 0;
        end else begin
            if (result_valid && result_ready) result_valid <= 0;
            valid_pipe <= {valid_pipe[PRODUCT_STAGES-2:0], accept};
            last_pipe <= {last_pipe[PRODUCT_STAGES-2:0], accept && in_last};
            for (integer s = 1; s < PRODUCT_STAGES; s++) begin
                if (valid_pipe[s-1]) begin
                    mask_pipe[s] <= mask_pipe[s-1];
                    slots_pipe[s] <= slots_pipe[s-1];
                end
            end
            sum_valid <= pipe_valid;
            if (pipe_valid) sum_last <= pipe_last;
            if (accept) begin
                mask_pipe[0] <= in_mask; slots_pipe[0] <= in_slots;
                if (in_last) last_seen <= 1;
            end
            if (sum_valid) begin
                for (integer j = 0; j < 8; j++) accum[j] <= accum_next[j];
                if (sum_last) begin
                    active <= 0; result_valid <= 1;
                    result_mask <= jobs_q; result_tags <= tags_q; result_wave_tag <= wave_q;
                    for (integer j = 0; j < 8; j++)
                        result_values[j*32 +: 32] <= jobs_q[j] ? accum_next[j] : 32'd0;
                end
            end
            if (cmd_valid && cmd_ready) begin
                active <= 1; last_seen <= 0; jobs_q <= cmd_job_mask;
                tags_q <= cmd_job_tags; wave_q <= cmd_wave_tag;
                for (integer j = 0; j < 8; j++) accum[j] <= 0;
            end
        end
    end
endmodule
