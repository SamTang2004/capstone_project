// Synthesis declarations for the ASAP7 FakeRAM cells used by the deployment
// wrappers.  Functional simulation should substitute the behavioral models
// shipped with OpenROAD-flow-scripts.
(* blackbox *)
module fakeram_512x8 (
    output logic [7:0] rd_out,
    input  logic [8:0] addr_in,
    input  logic       we_in,
    input  logic [7:0] wd_in,
    input  logic       clk,
    input  logic       ce_in
);
endmodule

(* blackbox *)
module fakeram7_256x32 (
    output logic [31:0] rd_out,
    input  logic [7:0]  addr_in,
    input  logic        we_in,
    input  logic [31:0] wd_in,
    input  logic        clk,
    input  logic        ce_in
);
endmodule
