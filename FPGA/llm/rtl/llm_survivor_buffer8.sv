// Stable cross-packet packing, with per-job dispatch fences and wave flush.
// in_count must be 0..8. Each packet belongs to in_job_tag; lane payloads are opaque.
module llm_survivor_buffer8 #(
    parameter integer WIDTH = 8,
    parameter integer PAYLOAD_WIDTH = 32,
    parameter integer JOB_TAG_WIDTH = 32,
    parameter integer WAVE_TAG_WIDTH = 16
) (
    input logic clk, rst_n,
    input logic in_valid, output logic in_ready,
    input logic [3:0] in_count,
    input logic [8*WIDTH-1:0] in_values,
    input logic [8*PAYLOAD_WIDTH-1:0] in_payload,
    input logic [23:0] in_source_lanes,
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
    localparam integer ENTRY_WIDTH = WIDTH + PAYLOAD_WIDTH + 3 + JOB_TAG_WIDTH;
    logic [ENTRY_WIDTH-1:0] entries [0:15], entries_d [0:15];
    logic [JOB_TAG_WIDTH-1:0] done_tags [0:7];
    // Distance to each fence in the buffered survivor stream. Saturates at zero.
    logic [4:0] remaining [0:7];
    logic [2:0] done_rd, done_wr;
    logic flush_q;
    logic [WAVE_TAG_WIDTH-1:0] wave_tag_q;
    logic push, pop, done_pop, done_push, pressure_drain;
    logic [3:0] sent;
    logic [4:0] base;
    integer lane;

    assign done_valid = rst_n && pending_jobs != 0 && remaining[done_rd] == 0;
    assign done_job_tag = done_valid ? done_tags[done_rd] : '0;
    assign done_pop = done_valid && done_ready;
    // A full fence queue with an unfinished head must be able to drain <8 items.
    // While this output stalls, the unfinished head cannot pop, so valid is stable.
    assign pressure_drain = pending_jobs == 8 && remaining[done_rd] != 0 && occupancy != 0;
    assign out_valid = rst_n && (occupancy >= 8 || flush_q || pressure_drain);
    assign out_count = out_valid ? ((occupancy >= 8) ? 4'd8 : occupancy[3:0]) : 4'd0;
    assign out_wave_last = out_valid && flush_q && occupancy <= 8;
    assign out_wave_tag = out_wave_last ? wave_tag_q : '0;
    assign pop = out_valid && out_ready;
    // Do not mutate the offered packet while it is stalled. At most 7 retained
    // entries remain after a dispatch; accepting up to 8 requires <=15 slots.
    assign in_ready = rst_n && !flush_q && (!out_valid || out_ready) &&
                      (in_count <= 8) && (!in_job_last || pending_jobs < 8 || done_pop);
    assign push = in_valid && in_ready;
    assign done_push = push && in_job_last;

    always_comb begin
        out_mask = '0; out_values = '0; out_payload = '0;
        out_source_lanes = '0; out_job_tags = '0;
        for (integer i = 0; i < 8; i++) begin
            if (i < out_count) begin
                out_mask[i] = 1'b1;
                {out_job_tags[i*JOB_TAG_WIDTH +: JOB_TAG_WIDTH],
                 out_source_lanes[i*3 +: 3], out_payload[i*PAYLOAD_WIDTH +: PAYLOAD_WIDTH],
                 out_values[i*WIDTH +: WIDTH]} = entries[i];
            end
        end
        sent = pop ? out_count : 0;
        base = occupancy - sent;
        for (lane = 0; lane < 16; lane++) begin
            entries_d[lane] = '0;
            // A dispatch either removes exactly eight entries or empties the
            // buffer. Thus any surviving tail shifts by a fixed eight slots.
            if (lane < base) begin
                if (!pop) entries_d[lane] = entries[lane];
                else if (lane < 8) entries_d[lane] = entries[lane+8];
            end
            for (integer src = 0; src < 8; src++) begin
                if (push && src < in_count && base + 5'(src) == lane)
                    entries_d[lane] = entries_d[lane] | {in_job_tag, in_source_lanes[src*3 +: 3],
                        in_payload[src*PAYLOAD_WIDTH +: PAYLOAD_WIDTH], in_values[src*WIDTH +: WIDTH]};
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            occupancy <= 0; pending_jobs <= 0; done_rd <= 0; done_wr <= 0;
            flush_q <= 0; wave_tag_q <= 0;
            for (integer i = 0; i < 16; i++) entries[i] <= '0;
            for (integer i = 0; i < 8; i++) begin remaining[i] <= 0; done_tags[i] <= 0; end
        end else begin
            occupancy <= base + (push ? in_count : 0);
            for (integer i = 0; i < 16; i++) entries[i] <= entries_d[i];
            if (pop && out_wave_last) flush_q <= 0;
            if (push && in_wave_last) begin flush_q <= 1; wave_tag_q <= in_wave_tag; end
            for (integer i = 0; i < 8; i++) begin
                if (pop) remaining[i] <= (remaining[i] > sent) ? remaining[i] - sent : 0;
            end
            if (done_pop) done_rd <= done_rd + 1'b1;
            if (done_push) begin
                done_tags[done_wr] <= in_job_tag;
                remaining[done_wr] <= base + in_count;
                done_wr <= done_wr + 1'b1;
            end
            case ({done_push, done_pop})
                2'b10: pending_jobs <= pending_jobs + 1'b1;
                2'b01: pending_jobs <= pending_jobs - 1'b1;
                default: pending_jobs <= pending_jobs;
            endcase
        end
    end
endmodule
