// Elastic stable compaction tree followed by a registered eight-entry serializer.
module llm_pipelined_work_finder #(
    parameter integer SCAN_LANES = 64,
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
    localparam integer BANKS = SCAN_LANES/8;
    localparam integer LEVELS = $clog2(BANKS);
    localparam integer COUNT_WIDTH = $clog2(SCAN_LANES+1);
    // Padding to a power of two makes variable shifts operate on whole records.
    localparam integer RECORD_LOG = $clog2(8+PAYLOAD_WIDTH+INDEX_WIDTH);
    localparam integer RECORD_WIDTH = 1 << RECORD_LOG;
    wire [SCAN_LANES*RECORD_WIDTH-1:0] data [0:LEVELS];
    wire [COUNT_WIDTH-1:0] counts [0:LEVELS][0:BANKS-1];
    logic [LEVELS:0] valid;
    wire [LEVELS:0] ready;
    logic [META_WIDTH-1:0] meta [0:LEVELS];
    wire [BANKS-1:0] bank_ready;
    // Two frame slots cut the backwards ready chain at the external boundary.
    // Admission depends only on registered occupancy, never downstream ready.
    localparam integer FRAME_WIDTH = SCAN_LANES*(9+PAYLOAD_WIDTH)+4;
    logic [FRAME_WIDTH-1:0] frame_queue [0:1];
    // Avoid the direct asynchronous LUTRAM-output -> metadata FF hold path.
    (* ram_style = "registers" *) logic [META_WIDTH-1:0] frame_meta_queue [0:1];
    logic [1:0] queued;
    logic rd, wr;
    wire [SCAN_LANES-1:0] frame_mask;
    wire [3:0] frame_shift;
    wire [8*SCAN_LANES-1:0] frame_values;
    wire [PAYLOAD_WIDTH*SCAN_LANES-1:0] frame_payload;
    wire [META_WIDTH-1:0] frame_meta;
    assign {frame_mask, frame_shift, frame_values, frame_payload} = frame_queue[rd];
    assign frame_meta = frame_meta_queue[rd];
    wire push = in_valid && in_ready;
    wire accept = queued != 0 && (&bank_ready);
    always_ff @(posedge clk) begin
        if (!rst_n) begin queued<=0; rd<=0; wr<=0; end
        else begin
            case ({push,accept})
                2'b10: queued <= queued+1'b1;
                2'b01: queued <= queued-1'b1;
                default: queued <= queued;
            endcase
            if(push) begin
                frame_queue[wr] <= {in_mask,in_shift,in_values,in_payload};
                frame_meta_queue[wr] <= in_meta;
                wr <= ~wr;
            end
            if(accept) rd <= ~rd;
        end
    end
    logic [COUNT_WIDTH-1:0] offset;
    wire [COUNT_WIDTH-1:0] remaining = counts[LEVELS][0] - offset;
    wire last_chunk = remaining <= 8;
    logic output_valid_q;
    wire output_space = !output_valid_q || out_ready;
    assign out_valid = rst_n && output_valid_q;
    assign ready[LEVELS] = !valid[LEVELS] || (output_space && last_chunk);
    assign in_ready = rst_n && queued < 2;

    for (genvar b=0; b<BANKS; b++) begin : banks
        wire [3:0] count;
        wire [63:0] values;
        wire [8*PAYLOAD_WIDTH-1:0] payload;
        wire [23:0] sources;
        wire downstream_ready;
        if (LEVELS == 0) assign downstream_ready = output_space && last_chunk;
        else assign downstream_ready = ready[1];
        llm_work_finder8 #(.PAYLOAD_WIDTH(PAYLOAD_WIDTH)) finder (
            .clk, .rst_n, .in_valid(accept), .in_ready(bank_ready[b]),
            .in_mask(frame_mask[b*8+:8]), .in_shift(frame_shift),
            .in_values(frame_values[b*64+:64]), .in_payload(frame_payload[b*8*PAYLOAD_WIDTH+:8*PAYLOAD_WIDTH]),
            .out_valid(), .out_ready(downstream_ready), .out_count(count),
            .out_mask(), .out_keep_mask(), .out_values(values), .out_payload(payload), .out_source_lanes(sources)
        );
        assign counts[0][b] = COUNT_WIDTH'(count);
        for (genvar i=0; i<8; i++) begin : records
            // Unused records must be zero for concatenation by OR to be valid.
            assign data[0][(b*8+i)*RECORD_WIDTH+:RECORD_WIDTH] = i<count ?
                RECORD_WIDTH'({INDEX_WIDTH'(b*8) | INDEX_WIDTH'(sources[i*3+:3]),
                               payload[i*PAYLOAD_WIDTH+:PAYLOAD_WIDTH], values[i*8+:8]}) : '0;
        end
    end
    if (LEVELS>0) assign ready[0] = !valid[0] || ready[1];
    always_ff @(posedge clk) begin
        if (!rst_n) begin valid[0]<=0; meta[0]<=0; end
        else if (ready[0]) begin
            valid[0] <= accept;
            if (accept) meta[0] <= frame_meta;
        end
    end

    for (genvar level=1; level<=LEVELS; level++) begin : merge_levels
        localparam integer CHILD = 8 << (level-1);
        localparam integer GROUP = 2*CHILD;
        if (level<LEVELS) assign ready[level] = !valid[level] || ready[level+1];
        always_ff @(posedge clk) begin
            if (!rst_n) begin valid[level]<=0; meta[level]<=0; end
            else if (ready[level]) begin
                valid[level] <= valid[level-1];
                if (valid[level-1]) meta[level] <= meta[level-1];
            end
        end
        for (genvar n=0; n<SCAN_LANES/GROUP; n++) begin : nodes
            logic [GROUP*RECORD_WIDTH-1:0] merged_q;
            logic [COUNT_WIDTH-1:0] count_q;
            wire [GROUP*RECORD_WIDTH-1:0] left_data =
                {{CHILD*RECORD_WIDTH{1'b0}}, data[level-1][n*GROUP*RECORD_WIDTH+:CHILD*RECORD_WIDTH]};
            wire [GROUP*RECORD_WIDTH-1:0] right_data =
                {{CHILD*RECORD_WIDTH{1'b0}}, data[level-1][(n*GROUP+CHILD)*RECORD_WIDTH+:CHILD*RECORD_WIDTH]};
            wire [GROUP*RECORD_WIDTH-1:0] merged = left_data |
                (right_data << {counts[level-1][n*2], {RECORD_LOG{1'b0}}});
            always_ff @(posedge clk) begin
                if (!rst_n) begin merged_q<=0; count_q<=0; end
                else if (ready[level] && valid[level-1]) begin
                    merged_q <= merged;
                    count_q <= counts[level-1][n*2] + counts[level-1][n*2+1];
                end
            end
            assign data[level][n*GROUP*RECORD_WIDTH+:GROUP*RECORD_WIDTH] = merged_q;
            assign counts[level][n] = count_q;
        end
        for (genvar n=SCAN_LANES/GROUP; n<BANKS; n++) assign counts[level][n] = 0;
    end

    wire [8*RECORD_WIDTH-1:0] chunk = data[LEVELS] >> {offset, {RECORD_LOG{1'b0}}};
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            offset<=0; output_valid_q<=0; out_count<=0; out_values<=0;
            out_payload<=0; out_source_indices<=0; out_meta<=0; out_frame_last<=0;
        end else if (output_space) begin
            output_valid_q <= valid[LEVELS];
            if (valid[LEVELS]) begin
                out_count <= last_chunk ? 4'(remaining) : 4'd8;
                out_frame_last <= last_chunk;
                out_meta <= meta[LEVELS];
                for (integer i=0; i<8; i++) begin
                    out_values[i*8+:8] <= chunk[i*RECORD_WIDTH+:8];
                    out_payload[i*PAYLOAD_WIDTH+:PAYLOAD_WIDTH] <= chunk[i*RECORD_WIDTH+8+:PAYLOAD_WIDTH];
                    out_source_indices[i*INDEX_WIDTH+:INDEX_WIDTH] <= chunk[i*RECORD_WIDTH+8+PAYLOAD_WIDTH+:INDEX_WIDTH];
                end
                offset <= last_chunk ? '0 : offset + COUNT_WIDTH'(8);
            end
        end
    end
endmodule
