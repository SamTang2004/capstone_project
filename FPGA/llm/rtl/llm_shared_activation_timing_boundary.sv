// Timing fixture only: not a functional ready/valid transport wrapper.
module llm_shared_activation_timing_boundary (
    input logic clk,
    input logic [675:0] fixture_inputs,
    output logic [306:0] fixture_outputs
);
    (* DONT_TOUCH="TRUE" *) logic [675:0] launch;
    (* DONT_TOUCH="TRUE" *) logic [306:0] capture;
    logic rst_n,cmd_valid,cmd_ready,in_valid,in_ready,in_last;
    logic [3:0] cmd_rows,cmd_passes,cmd_shift;
    logic [511:0] in_values;
    logic [63:0] in_mask;
    logic req_valid,req_ready,req_last,rsp_valid,rsp_ready,rsp_last;
    logic [11:0] req_index;
    logic [7:0] req_activation,req_mask,rsp_activation,rsp_mask;
    logic [63:0] rsp_weights;
    logic result_valid,result_ready;
    logic [255:0] result_values;
    logic debug_scanning,debug_scan_accept,debug_descriptor_write,debug_issue;
    logic [12:0] debug_descriptor_count;
    assign {rst_n,cmd_valid,cmd_rows,cmd_passes,cmd_shift,in_valid,in_values,
            in_mask,in_last,req_ready,rsp_valid,rsp_activation,rsp_mask,rsp_weights,
            rsp_last,result_ready}=launch;
    always_ff @(posedge clk) begin
        launch<=fixture_inputs;
        capture<={cmd_ready,in_ready,req_valid,req_index,req_activation,req_mask,
                  req_last,rsp_ready,result_valid,result_values,debug_scanning,
                  debug_scan_accept,debug_descriptor_write,debug_descriptor_count,debug_issue};
    end
    assign fixture_outputs=capture;
    (* KEEP_HIERARCHY="yes" *) llm_shared_activation8 #(.BITMAP(1),.STREAMING(1)) dut (.*);
endmodule
