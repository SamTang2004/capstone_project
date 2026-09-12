// Experimental independent-dot packing on the existing broadcast MAC array.
// Lane i owns PE[i,i] for the entire command. Cross products are discarded.
// Operand compaction is upstream; this block does not discover sparsity.
module llm_diagonal_tile (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [15:0] cmd_batch, cmd_k,
    input logic [31:0] cmd_tile_id,
    input logic [3:0] cmd_lanes,
    input logic cmd_w4,
    input logic [255:0] cmd_lane_tags,
    input logic operand_valid, output logic operand_ready,
    input logic [63:0] operand_a, operand_b,
    output logic completion_valid, input logic completion_ready,
    output logic [15:0] completion_batch,
    output logic [31:0] completion_tile_id, execution_cycles, input_wait_cycles,
    output logic [1:0] completion_status,
    output logic [7:0] result_mask,
    output logic [255:0] result_tags, result_values,
    output logic [2047:0] debug_accumulators
);
    logic [3:0] lanes_r;
    logic [255:0] tags_r;
    always_ff @(posedge clk) begin
        if (!rst_n) begin lanes_r <= 0; tags_r <= 0; end
        else if (cmd_valid && cmd_ready) begin
            lanes_r <= cmd_lanes;
            tags_r <= cmd_lane_tags;
        end
    end
    llm_tagged_tile tile (
        .clk, .rst_n, .cmd_valid, .cmd_ready, .cmd_batch,
        .cmd_task(cmd_tile_id), .cmd_w4, .cmd_k,
        .cmd_rows(cmd_lanes), .cmd_cols(cmd_lanes),
        .operand_valid, .operand_ready, .operand_a, .operand_b,
        .completion_valid, .completion_ready, .completion_batch,
        .completion_task(completion_tile_id), .completion_status,
        .execution_cycles, .input_wait_cycles,
        .completion_accumulators(debug_accumulators)
    );
    always_comb begin
        result_mask = 0; result_tags = 0; result_values = 0;
        for (int lane=0; lane<8; lane++) begin
            if (completion_valid && completion_status==0 && lane<lanes_r) begin
                result_mask[lane] = 1'b1;
                result_tags[lane*32+:32] = tags_r[lane*32+:32];
                result_values[lane*32+:32] = debug_accumulators[(lane*8+lane)*32+:32];
            end
        end
    end
endmodule
