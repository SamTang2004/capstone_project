// Wide candidate scan feeding the unchanged eight-DSP dot core.
module llm_wide_sparse_dot8 #(parameter integer SCAN_LANES = 32, parameter bit MERGE_BANKS = 0, parameter bit PIPELINE_MERGE = 0) (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [7:0] cmd_job_mask,
    input logic [255:0] cmd_job_tags,
    input logic [15:0] cmd_wave_tag,
    input logic cmd_w4,
    input logic in_valid, output logic in_ready,
    input logic [SCAN_LANES-1:0] in_mask,
    input logic [8*SCAN_LANES-1:0] in_a, in_b,
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
    logic [8*SCAN_LANES-1:0] weights;
    logic [SCAN_LANES-1:0] partner_mask;
    logic finder_ready, found_valid, found_ready, frame_last;
    logic [3:0] found_count;
    logic [63:0] found_a, found_b;
    logic [4:0] found_meta;
    logic core_valid, core_ready, core_last;
    logic [63:0] core_a, core_b;
    logic [23:0] core_slots;
    logic [7:0] core_mask;
    wire allowed = accepting && (jobs_q[in_job_slot] || (!in_mask && in_wave_last && !in_job_last));
    assign in_ready = rst_n && allowed && finder_ready;
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
        weights = 0; partner_mask = 0;
        for (integer i = 0; i < SCAN_LANES; i++) begin
            weights[i*8 +: 8] = w4_q ? {{4{in_b[i*8+3]}}, in_b[i*8 +: 4]} : in_b[i*8 +: 8];
            partner_mask[i] = in_mask[i] && weights[i*8 +: 8] != 0;
        end
    end
    llm_wide_work_finder #(.SCAN_LANES(SCAN_LANES), .MERGE_BANKS(MERGE_BANKS), .PIPELINE_MERGE(PIPELINE_MERGE)) finder (
        .clk, .rst_n, .in_valid(in_valid && allowed), .in_ready(finder_ready),
        .in_mask(partner_mask), .in_shift, .in_values(in_a), .in_payload(weights),
        .in_meta({in_job_slot, in_job_last, in_wave_last}),
        .out_valid(found_valid), .out_ready(found_ready), .out_count(found_count),
        .out_values(found_a), .out_payload(found_b), .out_source_indices(),
        .out_meta(found_meta), .out_frame_last(frame_last)
    );
    llm_survivor_buffer8 #(.PAYLOAD_WIDTH(8), .JOB_TAG_WIDTH(3)) buffer8 (
        .clk, .rst_n, .in_valid(found_valid), .in_ready(found_ready), .in_count(found_count),
        .in_values(found_a), .in_payload(found_b), .in_source_lanes(24'b0),
        .in_job_tag(found_meta[4:2]), .in_job_last(found_meta[1] && frame_last),
        .in_wave_last(found_meta[0] && frame_last), .in_wave_tag(wave_q),
        .out_valid(core_valid), .out_ready(core_ready), .out_count(), .out_mask(core_mask),
        .out_values(core_a), .out_payload(core_b), .out_source_lanes(), .out_job_tags(core_slots),
        .out_wave_last(core_last), .out_wave_tag(),
        .done_valid(), .done_ready(1'b1), .done_job_tag(), .occupancy(), .pending_jobs()
    );
    llm_dot_core8 core (
        .clk, .rst_n, .cmd_valid, .cmd_ready, .cmd_job_mask, .cmd_job_tags, .cmd_wave_tag,
        .in_valid(core_valid), .in_ready(core_ready), .in_mask(core_mask), .in_a(core_a), .in_b(core_b),
        .in_slots(core_slots), .in_last(core_last),
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
