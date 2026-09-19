// Eight-byte producer with magnitude, threshold, and occupancy stages.
module llm_activation_block_map (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [3:0] cmd_shift,
    input logic in_valid, output logic in_ready,
    input logic [63:0] in_values,
    input logic [7:0] in_mask,
    input logic in_last,
    output logic out_valid, input logic out_ready,
    output logic [63:0] out_blocks
);
    logic active,accepting;
    logic [7:0] threshold_bits,mask_q;
    logic [63:0] magnitudes,occupancy;
    logic [8:0] beat,beat_q,beat_qq;
    logic valid_q,valid_qq,last_q,last_qq,any_keep_q,block_keep;
    logic [7:0] kept;
    always_comb begin
        for(integer i=0;i<8;i++) kept[i]=mask_q[i] && (|(magnitudes[i*8+:8] & threshold_bits));
    end
    assign cmd_ready=rst_n && !active && !out_valid;
    assign in_ready=rst_n && accepting;
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            active<=0;accepting<=0;threshold_bits<=0;mask_q<=0;magnitudes<=0;
            beat<=0;beat_q<=0;beat_qq<=0;valid_q<=0;valid_qq<=0;
            last_q<=0;last_qq<=0;any_keep_q<=0;block_keep<=0;
            occupancy<=0;out_valid<=0;out_blocks<=0;
        end else begin
            if(out_valid && out_ready)out_valid<=0;
            valid_q<=in_valid && in_ready;valid_qq<=valid_q;
            if(cmd_valid && cmd_ready) begin
                active<=1;accepting<=1;threshold_bits<=8'hff<<cmd_shift;
                beat<=0;occupancy<=0;block_keep<=0;
            end
            if(in_valid && in_ready) begin
                for(integer i=0;i<8;i++)
                    magnitudes[i*8+:8]<=in_values[i*8+7] ? (~in_values[i*8+:8]+8'd1) : in_values[i*8+:8];
                mask_q<=in_mask;beat_q<=beat;last_q<=in_last;beat<=beat+1'b1;
                if(in_last)accepting<=0;
            end
            if(valid_q) begin
                any_keep_q<=|kept;beat_qq<=beat_q;last_qq<=last_q;
            end
            if(valid_qq) begin
                block_keep<=beat_qq[2:0]==7 ? 1'b0 : block_keep|any_keep_q;
                for(integer b=0;b<64;b++) begin
                    if(beat_qq[8:3]==b && (beat_qq[2:0]==7 || last_qq))
                        occupancy[b]<=block_keep|any_keep_q;
                    if(last_qq) out_blocks[b]<=occupancy[b] | ((beat_qq[8:3]==b) && (block_keep|any_keep_q));
                end
                if(last_qq) begin active<=0;out_valid<=1;end
            end
        end
    end
endmodule
