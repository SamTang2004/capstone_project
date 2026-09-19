// Stream the first eight-row tile while caching selected activations; replay
// across later tiles. Optional producer block map skips activation-memory reads.
// Both memory interfaces are ordered and must return their request sideband.
module llm_tiled_activation8 (
    input logic clk,rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [6:0] cmd_rows,cmd_blocks,
    input logic [3:0] cmd_shift,
    input logic cmd_skip,cmd_reuse,
    input logic [63:0] cmd_block_map,
    output logic a_req_valid, input logic a_req_ready,
    output logic [5:0] a_req_block,
    output logic a_req_last,
    input logic a_rsp_valid, output logic a_rsp_ready,
    input logic [5:0] a_rsp_block,
    input logic a_rsp_last,
    input logic [511:0] a_rsp_values,
    input logic [63:0] a_rsp_mask,
    output logic req_valid, input logic req_ready,
    output logic [11:0] req_index,
    output logic [2:0] req_tile,
    output logic [7:0] req_activation,req_mask,
    output logic req_last,
    input logic rsp_valid, output logic rsp_ready,
    input logic [7:0] rsp_activation,rsp_mask,
    input logic [63:0] rsp_weights,
    input logic rsp_last,
    output logic result_valid, input logic result_ready,
    output logic [255:0] result_values,
    output logic [2:0] result_tile,
    output logic [7:0] result_mask,
    output logic result_last,
    output logic debug_scanning,debug_selected,debug_cache_write,debug_cache_read
);
    typedef enum logic [2:0] {IDLE,START,SCAN,REPLAY,DRAIN} state_t;
    state_t state;
    logic reuse,replaying,empty_pending;
    logic [3:0] shift;
    logic [6:0] total_rows;
    logic [2:0] tile,last_tile;
    logic [63:0] block_map;
    logic [7:0] request_groups,request_lanes;
    logic [2:0] request_group;
    logic [7:0] rows_mask;
    logic [12:0] count,read_pos;
    logic [19:0] stream_descriptor,read_descriptor;
    (* ram_style="block" *) logic [19:0] cache [0:4095];
    logic write_q;
    logic [11:0] write_address;
    logic [19:0] write_data;
    wire [63:0] command_map;
    for(genvar b=0;b<64;b++) begin
        assign command_map[b]=(cmd_blocks>b) && (!cmd_skip || cmd_block_map[b]);
    end
    function automatic [2:0] first8(input logic [7:0] bits);
        begin first8=0;for(integer i=7;i>=0;i--) if(bits[i]) first8=3'(i);end
    endfunction
    always_comb begin
        for(integer r=0;r<8;r++) rows_mask[r]=({tile,3'b0}+7'(r))<total_rows;
    end
    wire [2:0] next_group=first8(request_groups);
    wire [2:0] next_lane=first8(request_lanes);
    wire [7:0] remaining_lanes=request_lanes & (request_lanes-8'd1);
    wire found_valid,found_ready,finder_ready,frame_last;
    wire [3:0] found_count;
    wire [63:0] found_values;
    wire [47:0] found_indices;
    wire [12:0] found_meta;
    wire req_space=!req_valid || req_ready;
    wire process_found=state==SCAN && found_valid && req_space;
    wire vector_done=frame_last && found_meta[12];
    wire [19:0] current_descriptor={12'(found_meta[11:0]+found_indices[5:0]),found_values[7:0]};
    assign found_ready=state==SCAN && req_space;
    assign a_rsp_ready=rst_n && state==SCAN && !empty_pending && finder_ready;
    llm_bitmap_finder64 #(.META_WIDTH(13)) finder (
        .clk,.rst_n,.in_valid(state==SCAN && (empty_pending || a_rsp_valid)),.in_ready(finder_ready),
        .in_mask(empty_pending ? 64'b0 : a_rsp_mask),
        .in_values(empty_pending ? 512'b0 : a_rsp_values),.in_shift(shift),
        .in_meta(empty_pending ? 13'h1000 : {a_rsp_last,a_rsp_block,6'b0}),
        .out_valid(found_valid),.out_ready(found_ready),.out_count(found_count),
        .out_values(found_values),.out_source_indices(found_indices),.out_meta(found_meta),.out_frame_last(frame_last)
    );
    wire core_cmd_ready,core_ready;
    assign cmd_ready=rst_n && state==IDLE;
    assign rsp_ready=rst_n && (state==SCAN || state==REPLAY || state==DRAIN) && core_ready;
    assign req_tile=tile;
    assign result_tile=tile;
    assign result_last=tile==last_tile;
    assign {req_index,req_activation}=req_mask==0 ? 20'b0 : replaying ? read_descriptor : stream_descriptor;
    assign debug_scanning=state==SCAN;
    assign debug_selected=process_found && found_count!=0;
    assign debug_cache_write=debug_selected && reuse && tile==0;
    assign debug_cache_read=state==REPLAY && req_space && read_pos<count;
    llm_dot_core8 #(.FIXED_LANE_SLOTS(1)) core (
        .clk,.rst_n,.cmd_valid(state==START),.cmd_ready(core_cmd_ready),
        .cmd_job_mask(rows_mask),.cmd_job_tags(256'b0),.cmd_wave_tag(16'b0),
        .in_valid(rsp_valid && (state==SCAN || state==REPLAY || state==DRAIN)),.in_ready(core_ready),
        .in_mask(rsp_mask),.in_a({8{rsp_activation}}),.in_b(rsp_weights),
        .in_slots({3'd7,3'd6,3'd5,3'd4,3'd3,3'd2,3'd1,3'd0}),.in_last(rsp_last),
        .result_valid,.result_ready,.result_mask,.result_tags(),.result_values,.result_wave_tag()
    );
    always_ff @(posedge clk) begin
        if(write_q) cache[write_address]<=write_data;
        write_q<=rst_n && debug_cache_write;
        if(debug_cache_write) begin write_address<=count[11:0];write_data<=current_descriptor;end
        // Hold the descriptor only when the output is stalled. Bounds/state
        // control validity separately; removing them from BRAM enable avoids
        // a count comparison on the memory setup path. Speculative reads are
        // unobservable outside REPLAY and never change a held request.
        if(req_space) read_descriptor<=cache[read_pos[11:0]];
        if(!rst_n) begin
            state<=IDLE;reuse<=0;replaying<=0;shift<=0;total_rows<=0;
            tile<=0;last_tile<=0;block_map<=0;empty_pending<=0;
            request_groups<=0;request_lanes<=0;request_group<=0;
            a_req_valid<=0;a_req_block<=0;a_req_last<=0;
            count<=0;read_pos<=0;req_valid<=0;req_mask<=0;req_last<=0;stream_descriptor<=0;
        end else begin
            if(req_valid && req_ready) begin req_valid<=0;if(req_last)state<=DRAIN;end
            if(a_req_valid && a_req_ready)a_req_valid<=0;
            // Separate group selection from the local eight-bit lane selector.
            // A registered output preserves the complete request under stalls.
            if(state==SCAN) begin
                if(request_lanes==0 && request_groups!=0) begin
                    request_group<=next_group;
                    request_lanes<=block_map[next_group*8+:8];
                    request_groups<=request_groups & (request_groups-8'd1);
                end else if(request_lanes!=0 && (!a_req_valid || a_req_ready)) begin
                    a_req_valid<=1;a_req_block<={request_group,next_lane};
                    a_req_last<=remaining_lanes==0 && request_groups==0;
                    request_lanes<=remaining_lanes;
                end
            end
            case(state)
                IDLE: if(cmd_valid && cmd_ready) begin
                    state<=START;reuse<=cmd_reuse;shift<=cmd_shift;total_rows<=cmd_rows;
                    tile<=0;last_tile<=3'((cmd_rows-1'b1)>>3);block_map<=command_map;
                    // START derives emptiness from the registered map. Avoid
                    // command decode plus a 64-bit reduction in one cycle.
                    empty_pending<=0;replaying<=0;count<=0;read_pos<=0;
                end
                START: if(core_cmd_ready) begin
                    replaying<=reuse && tile!=0;
                    if(reuse && tile!=0) begin state<=REPLAY;read_pos<=0;end
                    else begin
                        state<=SCAN;count<=0;empty_pending<=block_map==0;request_lanes<=0;
                        for(integer g=0;g<8;g++) request_groups[g]<=|block_map[g*8+:8];
                    end
                end
                SCAN: begin
                    if(empty_pending && finder_ready)empty_pending<=0;
                    if(process_found) begin
                        if(found_count!=0)count<=count+1'b1;
                        if(found_count!=0 || vector_done) begin
                            stream_descriptor<=current_descriptor;req_mask<=found_count!=0 ? rows_mask : 8'b0;
                            req_last<=vector_done;req_valid<=1;
                        end
                        if(vector_done)state<=DRAIN;
                    end
                end
                REPLAY: if(req_space) begin
                    if(read_pos<count) begin
                        req_mask<=rows_mask;req_last<=read_pos+1'b1==count;req_valid<=1;read_pos<=read_pos+1'b1;
                    end else if(count==0 && read_pos==0) begin
                        req_mask<=0;req_last<=1;req_valid<=1;read_pos<=1;
                    end
                end
                DRAIN: if(result_valid && result_ready) begin
                    if(tile==last_tile)state<=IDLE;
                    else begin tile<=tile+1'b1;state<=START;end
                end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
