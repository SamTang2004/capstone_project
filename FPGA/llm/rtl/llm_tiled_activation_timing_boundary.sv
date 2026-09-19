// Timing fixture only; not a functional memory/transport adapter.
module llm_tiled_activation_timing_boundary (
    input logic clk,
    input logic [754:0] fixture_inputs,
    output logic [316:0] fixture_outputs
);
    (* DONT_TOUCH="TRUE" *) logic [754:0] launch;
    (* DONT_TOUCH="TRUE" *) logic [316:0] capture;
    logic rst_n;
    logic cmd_valid;
    logic [6:0] cmd_rows;
    logic [6:0] cmd_blocks;
    logic [3:0] cmd_shift;
    logic cmd_skip;
    logic cmd_reuse;
    logic [63:0] cmd_block_map;
    logic a_req_ready;
    logic a_rsp_valid;
    logic [5:0] a_rsp_block;
    logic a_rsp_last;
    logic [511:0] a_rsp_values;
    logic [63:0] a_rsp_mask;
    logic req_ready;
    logic rsp_valid;
    logic [7:0] rsp_activation;
    logic [7:0] rsp_mask;
    logic [63:0] rsp_weights;
    logic rsp_last;
    logic result_ready;
    logic cmd_ready;
    logic a_req_valid;
    logic [5:0] a_req_block;
    logic a_req_last;
    logic a_rsp_ready;
    logic req_valid;
    logic [11:0] req_index;
    logic [2:0] req_tile;
    logic [7:0] req_activation;
    logic [7:0] req_mask;
    logic req_last;
    logic rsp_ready;
    logic result_valid;
    logic [255:0] result_values;
    logic [2:0] result_tile;
    logic [7:0] result_mask;
    logic result_last;
    logic debug_scanning;
    logic debug_selected;
    logic debug_cache_write;
    logic debug_cache_read;
    assign {rst_n,cmd_valid,cmd_rows,cmd_blocks,cmd_shift,cmd_skip,cmd_reuse,cmd_block_map,a_req_ready,a_rsp_valid,a_rsp_block,a_rsp_last,a_rsp_values,a_rsp_mask,req_ready,rsp_valid,rsp_activation,rsp_mask,rsp_weights,rsp_last,result_ready}=launch;
    always_ff @(posedge clk) begin
        launch<=fixture_inputs;
        capture<={cmd_ready,a_req_valid,a_req_block,a_req_last,a_rsp_ready,req_valid,req_index,req_tile,req_activation,req_mask,req_last,rsp_ready,result_valid,result_values,result_tile,result_mask,result_last,debug_scanning,debug_selected,debug_cache_write,debug_cache_read};
    end
    assign fixture_outputs=capture;
    (* KEEP_HIERARCHY="yes" *) llm_tiled_activation8 dut(.*);
endmodule
