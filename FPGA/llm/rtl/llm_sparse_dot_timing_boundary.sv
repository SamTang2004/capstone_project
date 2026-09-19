// TIMING HARNESS ONLY: explicit source/capture flops around every DUT port.
// The vectors are fixture controls/observations, not a ready/valid system API.
// Functional traffic correctness is verified on the DUT by the XSim scoreboard.
module llm_sparse_dot_timing_boundary #(
    parameter integer SCAN_LANES = 64,
    parameter bit PIPELINE_MERGE = 1,
    parameter bit MERGE_BANKS = 1,
    parameter integer INPUT_WIDTH = 294+17*SCAN_LANES,
    parameter integer OUTPUT_WIDTH = 556
) (
    input logic clk,
    input logic [INPUT_WIDTH-1:0] fixture_inputs,
    output wire [OUTPUT_WIDTH-1:0] fixture_outputs
);
    (* DONT_TOUCH = "true" *) logic [INPUT_WIDTH-1:0] launch_q;
    (* DONT_TOUCH = "true" *) logic [OUTPUT_WIDTH-1:0] capture_q;
    wire [OUTPUT_WIDTH-1:0] response;
    wire rst_n, cmd_valid, cmd_ready, cmd_w4, in_valid, in_ready;
    wire [7:0] cmd_job_mask;
    wire [255:0] cmd_job_tags;
    wire [15:0] cmd_wave_tag;
    wire [SCAN_LANES-1:0] in_mask;
    wire [8*SCAN_LANES-1:0] in_a, in_b;
    wire [3:0] in_shift;
    wire [2:0] in_job_slot;
    wire in_job_last, in_wave_last, result_valid, result_ready, debug_issue;
    wire [7:0] result_mask, debug_issue_mask, debug_useful_mask;
    wire [255:0] result_tags, result_values;
    wire [15:0] result_wave_tag;
    assign {rst_n,cmd_valid,cmd_job_mask,cmd_job_tags,cmd_wave_tag,cmd_w4,
            in_valid,in_mask,in_a,in_b,in_shift,in_job_slot,in_job_last,in_wave_last,result_ready} = launch_q;
    assign response = {cmd_ready,in_ready,result_valid,result_mask,result_tags,result_values,
                       result_wave_tag,debug_issue,debug_issue_mask,debug_useful_mask};
    assign fixture_outputs = capture_q;
    always_ff @(posedge clk) begin
        launch_q <= fixture_inputs;
        capture_q <= response;
    end
    (* KEEP_HIERARCHY = "yes" *) llm_wide_sparse_dot8 #(.SCAN_LANES(SCAN_LANES),
        .MERGE_BANKS(MERGE_BANKS), .PIPELINE_MERGE(PIPELINE_MERGE)) dut (.*);
endmodule
