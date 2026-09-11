module bcuda_asap7_sram_1rw_40 #(
    parameter int DEPTH = 256,
    parameter int ADDR_WIDTH = DEPTH <= 1 ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk,
    input  logic                  write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [39:0]           write_data,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [39:0]           read_data
);
    bcuda_asap7_sram_1rw_32 #(.DEPTH(DEPTH), .ADDR_WIDTH(ADDR_WIDTH)) low_sram (
        .clk, .write_enable, .write_addr, .write_data(write_data[31:0]),
        .read_addr, .read_data(read_data[31:0])
    );
    bcuda_asap7_sram_1rw_8 #(.DEPTH(DEPTH), .ADDR_WIDTH(ADDR_WIDTH)) high_sram (
        .clk, .write_enable, .write_addr, .write_data(write_data[39:32]),
        .read_addr, .read_data(read_data[39:32])
    );
endmodule
