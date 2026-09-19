// Registered group selection and local eight-lane selection for one survivor/cycle.
// Empty groups are skipped; each group transition takes one preparation cycle.
module llm_bitmap_finder64 #(parameter integer META_WIDTH = 13) (
    input logic clk, rst_n,
    input logic in_valid, output logic in_ready,
    input logic [63:0] in_mask,
    input logic [511:0] in_values,
    input logic [3:0] in_shift,
    input logic [META_WIDTH-1:0] in_meta,
    output logic out_valid, input logic out_ready,
    output logic [3:0] out_count,
    output logic [63:0] out_values,
    output logic [47:0] out_source_indices,
    output logic [META_WIDTH-1:0] out_meta,
    output logic out_frame_last
);
    typedef enum logic [2:0] {IDLE,FILTER,PREP,GROUP,EMIT} state_t;
    state_t state;
    logic [511:0] values,magnitudes;
    logic [63:0] mask,keep;
    logic [7:0] threshold_bits,groups,lanes;
    logic [63:0] group_values;
    logic [2:0] group_index;
    logic [META_WIDTH-1:0] meta;
    function automatic [2:0] first8(input logic [7:0] bits);
        begin first8=0;for(integer i=7;i>=0;i--) if(bits[i]) first8=3'(i);end
    endfunction
    wire [2:0] next_group=first8(groups);
    wire [2:0] lane_index=first8(lanes);
    wire [7:0] remaining_lanes=lanes & (lanes-8'd1);
    wire space=!out_valid || out_ready;
    assign in_ready=rst_n && state==IDLE;
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;values<=0;magnitudes<=0;mask<=0;keep<=0;
            threshold_bits<=0;groups<=0;lanes<=0;group_values<=0;group_index<=0;meta<=0;
            out_valid<=0;out_count<=0;out_values<=0;out_source_indices<=0;out_meta<=0;out_frame_last<=0;
        end else begin
            if(out_valid && out_ready)out_valid<=0;
            case(state)
                IDLE: if(in_valid && in_ready) begin
                    values<=in_values;mask<=in_mask;meta<=in_meta;
                    threshold_bits<=8'hff<<in_shift;
                    for(integer i=0;i<64;i++) begin
                        magnitudes[i*8+:8]<=in_values[i*8+7] ? (~in_values[i*8+:8]+8'd1) : in_values[i*8+:8];
                    end
                    state<=FILTER;
                end
                FILTER: begin
                    for(integer i=0;i<64;i++) keep[i]<=mask[i] && (|(magnitudes[i*8+:8] & threshold_bits));
                    state<=PREP;
                end
                PREP: begin
                    for(integer g=0;g<8;g++) groups[g]<=|keep[g*8+:8];
                    state<=GROUP;
                end
                GROUP: begin
                    if(groups!=0) begin
                        group_index<=next_group;lanes<=keep[next_group*8+:8];
                        group_values<=values[next_group*64+:64];
                        groups<=groups & (groups-8'd1);state<=EMIT;
                    end else if(space) begin
                        out_valid<=1;out_count<=0;out_values<=0;out_source_indices<=0;
                        out_meta<=meta;out_frame_last<=1;state<=IDLE;
                    end
                end
                EMIT: if(space) begin
                    out_valid<=1;out_count<=1;
                    out_values<={56'b0,group_values[lane_index*8+:8]};
                    out_source_indices<={42'b0,group_index,lane_index};
                    out_meta<=meta;out_frame_last<=remaining_lanes==0 && groups==0;
                    lanes<=remaining_lanes;
                    if(remaining_lanes==0)state<=groups==0 ? IDLE : GROUP;
                end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
