// Timing fixture only: register boundaries around the metadata producer.
module llm_block_map_timing_boundary (
    input logic clk,
    input logic [80:0] fixture_inputs,
    output logic [66:0] fixture_outputs
);
    (* DONT_TOUCH="TRUE" *) logic [80:0] launch;
    (* DONT_TOUCH="TRUE" *) logic [66:0] capture;
    logic rst_n;
    logic cmd_valid;
    logic [3:0] cmd_shift;
    logic in_valid;
    logic [63:0] in_values;
    logic [7:0] in_mask;
    logic in_last;
    logic out_ready;
    logic cmd_ready;
    logic in_ready;
    logic out_valid;
    logic [63:0] out_blocks;
    assign {rst_n,cmd_valid,cmd_shift,in_valid,in_values,in_mask,in_last,out_ready}=launch;
    always_ff @(posedge clk) begin
        launch<=fixture_inputs;
        capture<={cmd_ready,in_ready,out_valid,out_blocks};
    end
    assign fixture_outputs=capture;
    (* KEEP_HIERARCHY="yes" *) llm_activation_block_map dut(.*);
endmodule
