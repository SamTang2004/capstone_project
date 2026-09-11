module matrix_array_8x8 #(
    parameter bit USE_DSP =
`ifdef BCUDA_ASIC
        1'b0
`else
        1'b1
`endif
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          clear,
    input  logic          step_valid,
    input  logic [63:0]   activation_vector,
    input  logic [63:0]   weight_vector,
    output logic [2047:0] accumulators
);
    logic signed [7:0] unused_a [0:7][0:7];
    logic signed [7:0] unused_b [0:7][0:7];

    for (genvar row = 0; row < 8; row++) begin : gen_rows
        for (genvar col = 0; col < 8; col++) begin : gen_cols
            pe_int8_mac #(.USE_DSP(USE_DSP)) pe (
                .clk(clk),
                .rst_n(rst_n),
                .clear(clear),
                .enable(step_valid),
                .a_in(activation_vector[row*8 +: 8]),
                .b_in(weight_vector[col*8 +: 8]),
                .a_out(unused_a[row][col]),
                .b_out(unused_b[row][col]),
                .acc_out(accumulators[(row*8+col)*32 +: 32])
            );
        end
    end
endmodule
