module bcuda_asap7_sram_1rw_masked_64 #(
    parameter int DEPTH = 512,
    parameter int ADDR_WIDTH = DEPTH <= 1 ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk,
    input  logic [7:0]            write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [63:0]           write_data,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [63:0]           read_data
);
    for (genvar lane = 0; lane < 8; lane++) begin : gen_byte_lane
        bcuda_asap7_sram_1rw_8 #(.DEPTH(DEPTH), .ADDR_WIDTH(ADDR_WIDTH)) lane_sram (
            .clk, .write_enable(write_enable[lane]), .write_addr,
            .write_data(write_data[lane*8 +: 8]), .read_addr,
            .read_data(read_data[lane*8 +: 8])
        );
    end
endmodule
