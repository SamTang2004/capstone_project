// Finder + cross-packet survivor buffer. Job/wave sidebands follow the exact
// same acceptance boundary as the finder's registered packet, including empties.
module llm_work_packer8 #(
    parameter integer WIDTH = 8,
    parameter integer PAYLOAD_WIDTH = 32,
    parameter integer JOB_TAG_WIDTH = 32,
    parameter integer WAVE_TAG_WIDTH = 16,
    parameter integer SIGNED_MAGNITUDE = 1,
    parameter integer SHIFT_WIDTH = $clog2(WIDTH + 1)
) (
    input logic clk, rst_n,
    input logic in_valid, output logic in_ready,
    input logic [7:0] in_mask,
    input logic [SHIFT_WIDTH-1:0] in_shift,
    input logic [8*WIDTH-1:0] in_values,
    input logic [8*PAYLOAD_WIDTH-1:0] in_payload,
    input logic [JOB_TAG_WIDTH-1:0] in_job_tag,
    input logic in_job_last, in_wave_last,
    input logic [WAVE_TAG_WIDTH-1:0] in_wave_tag,
    output logic out_valid, input logic out_ready,
    output logic [3:0] out_count,
    output logic [7:0] out_mask,
    output logic [8*WIDTH-1:0] out_values,
    output logic [8*PAYLOAD_WIDTH-1:0] out_payload,
    output logic [23:0] out_source_lanes,
    output logic [8*JOB_TAG_WIDTH-1:0] out_job_tags,
    output logic out_wave_last,
    output logic [WAVE_TAG_WIDTH-1:0] out_wave_tag,
    output logic done_valid, input logic done_ready,
    output logic [JOB_TAG_WIDTH-1:0] done_job_tag,
    output logic [4:0] occupancy,
    output logic [3:0] pending_jobs
);
    logic found_valid, found_ready;
    logic [3:0] found_count;
    logic [8*WIDTH-1:0] found_values;
    logic [8*PAYLOAD_WIDTH-1:0] found_payload;
    logic [23:0] found_sources;
    logic [JOB_TAG_WIDTH-1:0] job_tag_q;
    logic job_last_q, wave_last_q;
    logic [WAVE_TAG_WIDTH-1:0] wave_tag_q;
    llm_work_finder8 #(.WIDTH(WIDTH), .PAYLOAD_WIDTH(PAYLOAD_WIDTH),
        .SIGNED_MAGNITUDE(SIGNED_MAGNITUDE), .SHIFT_WIDTH(SHIFT_WIDTH)) finder (
        .clk, .rst_n, .in_valid, .in_ready, .in_mask, .in_shift, .in_values, .in_payload,
        .out_valid(found_valid), .out_ready(found_ready), .out_count(found_count),
        .out_mask(), .out_keep_mask(), .out_values(found_values),
        .out_payload(found_payload), .out_source_lanes(found_sources)
    );
    always_ff @(posedge clk) begin
        if (!rst_n) begin job_tag_q <= 0; job_last_q <= 0; wave_last_q <= 0; wave_tag_q <= 0; end
        else if (in_valid && in_ready) begin
            job_tag_q <= in_job_tag; job_last_q <= in_job_last;
            wave_last_q <= in_wave_last; wave_tag_q <= in_wave_tag;
        end
    end
    llm_survivor_buffer8 #(.WIDTH(WIDTH), .PAYLOAD_WIDTH(PAYLOAD_WIDTH),
        .JOB_TAG_WIDTH(JOB_TAG_WIDTH), .WAVE_TAG_WIDTH(WAVE_TAG_WIDTH)) buffer8 (
        .clk, .rst_n, .in_valid(found_valid), .in_ready(found_ready), .in_count(found_count),
        .in_values(found_values), .in_payload(found_payload), .in_source_lanes(found_sources),
        .in_job_tag(job_tag_q), .in_job_last(job_last_q), .in_wave_last(wave_last_q), .in_wave_tag(wave_tag_q),
        .out_valid, .out_ready, .out_count, .out_mask, .out_values, .out_payload,
        .out_source_lanes, .out_job_tags, .out_wave_last, .out_wave_tag,
        .done_valid, .done_ready, .done_job_tag, .occupancy, .pending_jobs
    );
endmodule
