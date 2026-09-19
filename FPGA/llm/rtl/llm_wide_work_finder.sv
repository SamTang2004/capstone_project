// Parallel eight-entry compactors, one captured frame, up to eight survivors out.
// SCAN_LANES is one of 8,16,32,64. Empty banks are skipped without drain cycles.
// MERGE_BANKS=1 selects up to eight survivors across bank boundaries each cycle.
// PIPELINE_MERGE=1 instead selects the buffered, pipelined merge tree.
// A frame emits a final chunk even if every entry is discarded. Metadata is
// repeated per chunk; out_frame_last identifies where end markers take effect.
module llm_wide_work_finder #(
    parameter integer SCAN_LANES = 32,
    parameter bit MERGE_BANKS = 0,
    parameter bit PIPELINE_MERGE = 0,
    parameter integer PAYLOAD_WIDTH = 8,
    parameter integer META_WIDTH = 5,
    parameter integer INDEX_WIDTH = $clog2(SCAN_LANES)
) (
    input logic clk, rst_n,
    input logic in_valid, output logic in_ready,
    input logic [SCAN_LANES-1:0] in_mask,
    input logic [3:0] in_shift,
    input logic [8*SCAN_LANES-1:0] in_values,
    input logic [PAYLOAD_WIDTH*SCAN_LANES-1:0] in_payload,
    input logic [META_WIDTH-1:0] in_meta,
    output logic out_valid, input logic out_ready,
    output logic [3:0] out_count,
    output logic [63:0] out_values,
    output logic [8*PAYLOAD_WIDTH-1:0] out_payload,
    output logic [8*INDEX_WIDTH-1:0] out_source_indices,
    output logic [META_WIDTH-1:0] out_meta,
    output logic out_frame_last
);
    if (PIPELINE_MERGE) begin : pipelined
        llm_pipelined_work_finder #(.SCAN_LANES(SCAN_LANES), .PAYLOAD_WIDTH(PAYLOAD_WIDTH),
            .META_WIDTH(META_WIDTH), .INDEX_WIDTH(INDEX_WIDTH)) finder (.*);
    end else begin : legacy
    localparam integer BANKS = SCAN_LANES / 8;
    wire [BANKS-1:0] bank_valid, bank_ready;
    logic [BANKS-1:0] bank_consume, nonempty;
    wire [3:0] counts [0:BANKS-1];
    wire [63:0] values [0:BANKS-1];
    wire [8*PAYLOAD_WIDTH-1:0] payloads [0:BANKS-1];
    wire [23:0] sources [0:BANKS-1];
    logic busy;
    logic [META_WIDTH-1:0] meta_q;
    logic picked;
    integer selected;
    wire accept = in_valid && in_ready;
    assign out_valid = rst_n && busy;
    assign in_ready = rst_n && (!busy || (out_ready && out_frame_last));
    assign out_meta = meta_q;

    for (genvar b = 0; b < BANKS; b++) begin : banks
        llm_work_finder8 #(.PAYLOAD_WIDTH(PAYLOAD_WIDTH)) finder (
            .clk, .rst_n, .in_valid(accept), .in_ready(bank_ready[b]),
            .in_mask(in_mask[b*8 +: 8]), .in_shift,
            .in_values(in_values[b*64 +: 64]), .in_payload(in_payload[b*8*PAYLOAD_WIDTH +: 8*PAYLOAD_WIDTH]),
            .out_valid(bank_valid[b]), .out_ready(bank_consume[b]), .out_count(counts[b]),
            .out_mask(), .out_keep_mask(), .out_values(values[b]),
            .out_payload(payloads[b]), .out_source_lanes(sources[b])
        );
    end
    if (!MERGE_BANKS) begin : bank_dispatch
    always_comb begin
        nonempty = 0; picked = 0; selected = 0;
        for (integer b = 0; b < BANKS; b++) begin
            nonempty[b] = bank_valid[b] && counts[b] != 0;
            if (nonempty[b] && !picked) begin picked = 1; selected = b; end
        end
        out_count = 0; out_values = 0; out_payload = 0; out_source_indices = 0;
        if (busy && picked) begin
            out_count = counts[selected]; out_values = values[selected]; out_payload = payloads[selected];
            for (integer i = 0; i < 8; i++)
                if (i < out_count)
                    out_source_indices[i*INDEX_WIDTH +: INDEX_WIDTH] = INDEX_WIDTH'(selected*8) |
                        INDEX_WIDTH'(sources[selected][i*3 +: 3]);
        end
        out_frame_last = busy && ((nonempty & (nonempty - 1'b1)) == 0);
        for (integer b = 0; b < BANKS; b++)
            bank_consume[b] = counts[b] == 0 || (busy && picked && b == selected && out_ready);
    end
    end else begin : merged_dispatch
        // Hold all local compactors until the frame drains. Prefix ranks place
        // their survivors in one stable sequence; offset selects its next eight.
        localparam integer RANK_WIDTH = $clog2(SCAN_LANES + 1);
        logic [RANK_WIDTH-1:0] offset;
        wire [RANK_WIDTH-1:0] prefix [0:BANKS];
        assign prefix[0] = 0;
        for (genvar b = 0; b < BANKS; b++) begin : ranks
            assign prefix[b+1] = prefix[b] + RANK_WIDTH'(counts[b]);
        end
        always_comb begin
            out_count = 0; out_values = 0; out_payload = 0; out_source_indices = 0;
            if (busy) begin
                out_count = prefix[BANKS] - offset >= 8 ? 4'd8 : 4'(prefix[BANKS] - offset);
                for (integer lane = 0; lane < 8; lane++) begin
                    for (integer b = 0; b < BANKS; b++) begin
                        for (integer i = 0; i < 8; i++) begin
                            if (i < counts[b] && lane < out_count &&
                                prefix[b] + RANK_WIDTH'(i) == offset + RANK_WIDTH'(lane)) begin
                                out_values[lane*8 +: 8] |= values[b][i*8 +: 8];
                                out_payload[lane*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] |= payloads[b][i*PAYLOAD_WIDTH +: PAYLOAD_WIDTH];
                                out_source_indices[lane*INDEX_WIDTH +: INDEX_WIDTH] |= INDEX_WIDTH'(b*8) | INDEX_WIDTH'(sources[b][i*3 +: 3]);
                            end
                        end
                    end
                end
            end
            out_frame_last = busy && prefix[BANKS] - offset <= 8;
            bank_consume = {BANKS{out_valid && out_ready && out_frame_last}};
        end
        always_ff @(posedge clk) begin
            if (!rst_n) offset <= 0;
            else if (accept) offset <= 0;
            else if (out_valid && out_ready) begin
                if (out_frame_last) offset <= 0;
                else offset <= offset + RANK_WIDTH'(8);
            end
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin busy <= 0; meta_q <= 0; end
        else begin
            if (out_valid && out_ready && out_frame_last) busy <= 0;
            if (accept) begin busy <= 1; meta_q <= in_meta; end
        end
    end
    end
endmodule
