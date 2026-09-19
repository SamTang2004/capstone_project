// MODE=0 masked dense, MODE=1 local finder, MODE=2 cross-packet packing.
// All modes compute the same thresholded dot products on the same eight DSPs.
// One wave has up to eight declared jobs; job slots index its external tag table.
module llm_sparse_dot8 #(parameter integer MODE = 2) (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [7:0] cmd_job_mask,
    input logic [255:0] cmd_job_tags,
    input logic [15:0] cmd_wave_tag,
    input logic cmd_w4,
    input logic in_valid, output logic in_ready,
    input logic [7:0] in_mask,
    input logic [63:0] in_a, in_b,
    input logic [3:0] in_shift,
    input logic [2:0] in_job_slot,
    input logic in_job_last, in_wave_last,
    output logic result_valid, input logic result_ready,
    output logic [7:0] result_mask,
    output logic [255:0] result_tags, result_values,
    output logic [15:0] result_wave_tag,
    output logic debug_issue,
    output logic [7:0] debug_issue_mask, debug_useful_mask
);
    logic accepting, w4_q;
    logic [7:0] jobs_q;
    logic [15:0] wave_q;
    logic [63:0] weights, masked_a;
    logic [7:0] partner_mask;
    logic front_ready, core_ready, core_valid, core_last;
    logic [7:0] core_mask;
    logic [63:0] core_a, core_b;
    logic [23:0] core_slots;
    logic [7:0] magnitude, raw_a;
    wire allowed = accepting && (jobs_q[in_job_slot] || (!in_mask && in_wave_last && !in_job_last));
    wire front_valid = in_valid && allowed;
    assign in_ready = rst_n && allowed && front_ready;
    always_ff @(posedge clk) begin
        if (!rst_n) begin accepting <= 0; jobs_q <= 0; wave_q <= 0; w4_q <= 0; end
        else begin
            if (cmd_valid && cmd_ready) begin
                accepting <= 1; jobs_q <= cmd_job_mask; wave_q <= cmd_wave_tag; w4_q <= cmd_w4;
            end
            if (in_valid && in_ready && in_wave_last) accepting <= 0;
        end
    end
    always_comb begin
        weights = 0; partner_mask = 0; masked_a = 0; raw_a = 0; magnitude = 0;
        for (integer i = 0; i < 8; i++) begin
            weights[i*8 +: 8] = w4_q ? {{4{in_b[i*8+3]}}, in_b[i*8 +: 4]} : in_b[i*8 +: 8];
            partner_mask[i] = in_mask[i] && weights[i*8 +: 8] != 0;
            raw_a = in_a[i*8 +: 8];
            magnitude = raw_a[7] ? (~raw_a + 1'b1) : raw_a;
            if (in_shift < 8 && (|(magnitude >> in_shift))) masked_a[i*8 +: 8] = raw_a;
        end
    end
    if (MODE == 0) begin : dense
        assign front_ready = core_ready;
        assign core_valid = front_valid;
        assign core_mask = in_mask;
        assign core_a = masked_a;
        assign core_b = weights;
        assign core_slots = {8{in_job_slot}};
        assign core_last = in_wave_last;
    end else if (MODE == 1) begin : local_only
        logic [2:0] slot_q;
        logic last_q;
        llm_work_finder8 #(.PAYLOAD_WIDTH(8)) finder (
            .clk, .rst_n, .in_valid(front_valid), .in_ready(front_ready), .in_mask(partner_mask),
            .in_shift, .in_values(in_a), .in_payload(weights),
            .out_valid(core_valid), .out_ready(core_ready), .out_count(), .out_mask(core_mask),
            .out_keep_mask(), .out_values(core_a), .out_payload(core_b), .out_source_lanes()
        );
        always_ff @(posedge clk) begin
            if (!rst_n) begin slot_q <= 0; last_q <= 0; end
            else if (front_valid && front_ready) begin slot_q <= in_job_slot; last_q <= in_wave_last; end
        end
        assign core_slots = {8{slot_q}};
        assign core_last = last_q;
    end else begin : cross_packet
        llm_work_packer8 #(.PAYLOAD_WIDTH(8), .JOB_TAG_WIDTH(3)) packer (
            .clk, .rst_n, .in_valid(front_valid), .in_ready(front_ready),
            .in_mask(partner_mask), .in_shift, .in_values(in_a), .in_payload(weights),
            .in_job_tag(in_job_slot), .in_job_last, .in_wave_last, .in_wave_tag(wave_q),
            .out_valid(core_valid), .out_ready(core_ready), .out_count(), .out_mask(core_mask),
            .out_values(core_a), .out_payload(core_b), .out_source_lanes(), .out_job_tags(core_slots),
            .out_wave_last(core_last), .out_wave_tag(),
            .done_valid(), .done_ready(1'b1), .done_job_tag(), .occupancy(), .pending_jobs()
        );
    end
    llm_dot_core8 core (
        .clk, .rst_n, .cmd_valid, .cmd_ready, .cmd_job_mask, .cmd_job_tags, .cmd_wave_tag,
        .in_valid(core_valid), .in_ready(core_ready), .in_mask(core_mask),
        .in_a(core_a), .in_b(core_b), .in_slots(core_slots), .in_last(core_last),
        .result_valid, .result_ready, .result_mask, .result_tags, .result_values, .result_wave_tag
    );
    assign debug_issue = core_valid && core_ready;
    assign debug_issue_mask = core_mask;
    always_comb begin
        debug_useful_mask = 0;
        for (integer i = 0; i < 8; i++)
            debug_useful_mask[i] = core_mask[i] && core_a[i*8 +: 8] != 0 && core_b[i*8 +: 8] != 0;
    end
endmodule
