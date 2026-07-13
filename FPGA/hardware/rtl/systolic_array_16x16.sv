module systolic_array_16x16 (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      clear,
    input  logic                      enable,
    input  logic signed [8*16-1:0]    a_west_flat,
    input  logic signed [8*16-1:0]    b_north_flat,
    output logic signed [32*256-1:0]  c_flat
);

    logic signed [7:0]  a_chain [0:15][0:15];
    logic signed [7:0]  b_chain [0:15][0:15];
    logic signed [31:0] acc     [0:15][0:15];

    genvar r;
    genvar c;

    generate
        for (r = 0; r < 16; r = r + 1) begin : gen_rows
            for (c = 0; c < 16; c = c + 1) begin : gen_cols
                // The Zynq-7020 has 220 DSP48s total. Reserve headroom for
                // the surrounding datapath and map 180 of 256 MACs to DSPs.
                pe_int8_mac #(.USE_DSP((r * 16 + c) < 180)) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .clear(clear),
                    .enable(enable),
                    .a_in(c == 0 ? a_west_flat[r*8 +: 8] : a_chain[r][c-1]),
                    .b_in(r == 0 ? b_north_flat[c*8 +: 8] : b_chain[r-1][c]),
                    .a_out(a_chain[r][c]),
                    .b_out(b_chain[r][c]),
                    .acc_out(acc[r][c])
                );

                assign c_flat[(r*16 + c)*32 +: 32] = acc[r][c];
            end
        end
    endgenerate

endmodule
