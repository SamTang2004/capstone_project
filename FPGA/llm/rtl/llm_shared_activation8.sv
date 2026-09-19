// Research prototype: prepare descriptors once, then broadcast each activation
// over up to eight output rows. Weight memory is an ordered external interface.
// It must retain req_activation/mask/last as response sideband; see testbench.
// cmd_passes > 1 is a repeated-finding ablation: resend the same vector each pass.
module llm_shared_activation8 #(
    parameter bit BITMAP = 0,
    parameter bit STREAMING = 0,
    parameter integer INDEX_WIDTH = 12
) (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [3:0] cmd_rows, cmd_passes, cmd_shift,
    input logic in_valid, output logic in_ready,
    input logic [511:0] in_values,
    input logic [63:0] in_mask,
    input logic in_last,
    output logic req_valid, input logic req_ready,
    output logic [INDEX_WIDTH-1:0] req_index,
    output logic [7:0] req_activation, req_mask,
    output logic req_last,
    input logic rsp_valid, output logic rsp_ready,
    input logic [7:0] rsp_activation, rsp_mask,
    input logic [63:0] rsp_weights,
    input logic rsp_last,
    output logic result_valid, input logic result_ready,
    output logic [255:0] result_values,
    output logic debug_scanning, debug_scan_accept, debug_descriptor_write,
    output logic [INDEX_WIDTH:0] debug_descriptor_count,
    output logic debug_issue
);
    localparam integer CAPACITY = 1 << INDEX_WIDTH;
    localparam integer META_WIDTH = INDEX_WIDTH+1;
    typedef enum logic [2:0] {IDLE, COLLECT, START, REPLAY, DRAIN} state_t;
    state_t state;
    logic [3:0] passes, shift;
    logic [7:0] rows_mask;
    logic input_closed;
    logic [INDEX_WIDTH-1:0] base;
    logic [INDEX_WIDTH:0] count, read_pos;
    // One write and one synchronous read; no reset of descriptor storage.
    (* ram_style = "block" *) logic [INDEX_WIDTH+7:0] descriptors [0:CAPACITY-1];
    logic [INDEX_WIDTH+7:0] descriptor_read;
    logic descriptor_write_q;
    logic [INDEX_WIDTH-1:0] descriptor_address_q;
    logic [INDEX_WIDTH+7:0] descriptor_data_q;
    logic [INDEX_WIDTH+7:0] streaming_descriptor;
    assign {req_index,req_activation} = req_mask == 0 ? '0 :
                                      STREAMING ? streaming_descriptor : descriptor_read;
    logic found_valid, found_ready, finder_ready, frame_last;
    logic [3:0] found_count, chunk_pos;
    logic [63:0] found_values;
    logic [47:0] found_indices;
    logic [META_WIDTH-1:0] found_meta;
    wire core_cmd_ready, core_in_ready;
    wire [META_WIDTH-1:0] meta = {in_last,base};
    wire finder_valid = in_valid && state == COLLECT && !input_closed;
    wire chunk_done = found_count == 0 || chunk_pos + 1'b1 == found_count;
    wire stream_active = STREAMING && passes == 1;
    wire collect_space = !stream_active || !req_valid || req_ready;
    wire process_found = state == COLLECT && found_valid && collect_space;
    wire write_descriptor = process_found && found_count != 0;
    wire vector_done = chunk_done && frame_last && found_meta[INDEX_WIDTH];
    wire [INDEX_WIDTH+7:0] current_descriptor = {
        INDEX_WIDTH'(found_meta[INDEX_WIDTH-1:0] + found_indices[chunk_pos*6+:6]),
        found_values[chunk_pos*8+:8]};
    assign found_ready = state == COLLECT && chunk_done && collect_space;
    assign in_ready = rst_n && state == COLLECT && !input_closed && finder_ready;
    assign cmd_ready = rst_n && state == IDLE && (!STREAMING || core_cmd_ready);
    assign debug_scanning = state == COLLECT;
    assign debug_scan_accept = in_valid && in_ready;
    assign debug_descriptor_write = write_descriptor;
    assign debug_descriptor_count = count;
    if (BITMAP) begin : bitmap
        llm_bitmap_finder64 #(.META_WIDTH(META_WIDTH)) finder (
            .clk,.rst_n,.in_valid(finder_valid),.in_ready(finder_ready),
            .in_mask,.in_values,.in_shift(shift),.in_meta(meta),
            .out_valid(found_valid),.out_ready(found_ready),.out_count(found_count),
            .out_values(found_values),.out_source_indices(found_indices),
            .out_meta(found_meta),.out_frame_last(frame_last)
        );
    end else begin : tree
        llm_pipelined_work_finder #(.PAYLOAD_WIDTH(1),.META_WIDTH(META_WIDTH)) finder (
            .clk,.rst_n,.in_valid(finder_valid),.in_ready(finder_ready),
            .in_mask,.in_values,.in_shift(shift),.in_payload(64'b0),.in_meta(meta),
            .out_valid(found_valid),.out_ready(found_ready),.out_count(found_count),
            .out_values(found_values),.out_payload(),.out_source_indices(found_indices),
            .out_meta(found_meta),.out_frame_last(frame_last)
        );
    end
    wire core_cmd_valid = STREAMING ? cmd_valid && state == IDLE : state == START;
    wire accepting_responses = state == REPLAY || state == DRAIN || (state == COLLECT && stream_active);
    assign rsp_ready = rst_n && accepting_responses && core_in_ready;
    assign debug_issue = rsp_valid && rsp_ready;
    wire [63:0] broadcast_a = {8{rsp_activation}};
    wire [23:0] slots = {3'd7,3'd6,3'd5,3'd4,3'd3,3'd2,3'd1,3'd0};
    llm_dot_core8 core (
        .clk,.rst_n,.cmd_valid(core_cmd_valid),.cmd_ready(core_cmd_ready),
        .cmd_job_mask(STREAMING && state==IDLE ? 8'((9'd1 << cmd_rows)-1'b1) : rows_mask),.cmd_job_tags(256'b0),.cmd_wave_tag(16'b0),
        .in_valid(rsp_valid && accepting_responses),.in_ready(core_in_ready),
        .in_mask(rsp_mask),.in_a(broadcast_a),.in_b(rsp_weights),.in_slots(slots),.in_last(rsp_last),
        .result_valid,.result_ready,.result_mask(),.result_tags(),.result_values,.result_wave_tag()
    );
    always_ff @(posedge clk) begin
        // Separate selection/address arithmetic from the BRAM write setup path.
        // START supplies the drain cycle before the first synchronous read.
        if (descriptor_write_q && !STREAMING)
            descriptors[descriptor_address_q] <= descriptor_data_q;
        descriptor_write_q <= rst_n && write_descriptor && !STREAMING;
        if (write_descriptor && !STREAMING) begin
            descriptor_address_q <= count[INDEX_WIDTH-1:0];
            descriptor_data_q <= current_descriptor;
        end
        // Keep the RAM read register free of resets and alternate data inputs
        // so Vivado can infer a true synchronous block RAM read port.
        if (!STREAMING && state == REPLAY && (!req_valid || req_ready) && read_pos < count)
            descriptor_read <= descriptors[read_pos[INDEX_WIDTH-1:0]];
        if (!rst_n) begin
            state<=IDLE; passes<=0; shift<=0; rows_mask<=0; base<=0;
            count<=0; read_pos<=0; input_closed<=0; chunk_pos<=0;
            req_valid<=0; req_mask<=0; req_last<=0;
            streaming_descriptor<=0;
        end else begin
            if (req_valid && req_ready) begin
                req_valid<=0;
                if (req_last) state<=DRAIN;
            end
            case (state)
                IDLE: if (cmd_valid && cmd_ready) begin
                    state<=COLLECT; passes<=cmd_passes; shift<=cmd_shift;
                    rows_mask<=8'((9'd1 << cmd_rows)-1'b1);
                    base<=0; count<=0; read_pos<=0; input_closed<=0; chunk_pos<=0;
                end
                COLLECT: begin
                    if (in_valid && in_ready) begin
                        base<=base+INDEX_WIDTH'(64);
                        if (in_last) input_closed<=1;
                    end
                    if (process_found) begin
                        if (found_count != 0) count<=count+1'b1;
                        if (stream_active && (found_count != 0 || vector_done)) begin
                            streaming_descriptor<=current_descriptor;
                            req_mask<=found_count!=0 ? rows_mask : 8'b0;
                            req_last<=vector_done;
                            req_valid<=1;
                        end
                        if (chunk_done) begin
                            chunk_pos<=0;
                            if (vector_done) begin
                                if (passes > 1) begin
                                    passes<=passes-1'b1; count<=0; base<=0; input_closed<=0;
                                end else state<=STREAMING ? DRAIN : START;
                            end
                        end else chunk_pos<=chunk_pos+1'b1;
                    end
                end
                START: if (core_cmd_ready) state<=REPLAY;
                REPLAY: if (!req_valid || req_ready) begin
                    if (read_pos < count) begin
                        req_mask<=rows_mask; req_last<=read_pos+1'b1==count;
                        req_valid<=1; read_pos<=read_pos+1'b1;
                    end else if (count==0 && read_pos==0) begin
                        req_mask<=0; req_last<=1;
                        req_valid<=1; read_pos<=1;
                    end
                end
                DRAIN: if (result_valid && result_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
