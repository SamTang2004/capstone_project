// ASAP7 deployment wrapper for a synchronous 64-bit, one-read/one-write
// logical memory.  The physical building block is the 128x64 single-port
// FakeRAM macro supplied with OpenROAD-flow-scripts.
//
// The scheduler keeps reads and writes in separate phases in this ASIC copy.
// If write_enable is asserted, the macro port services the write and read_data
// is not meaningful for that cycle.
module bcuda_asap7_sram_1rw_64 #(
    parameter int DEPTH = 128,
    parameter int ADDR_WIDTH = DEPTH <= 1 ? 1 : $clog2(DEPTH),
    localparam int MACRO_DEPTH = 128,
    localparam int BANK_COUNT = (DEPTH + MACRO_DEPTH - 1) / MACRO_DEPTH,
    localparam int BANK_WIDTH = BANK_COUNT <= 1 ? 1 : $clog2(BANK_COUNT)
) (
    input  logic                  clk,
    input  logic                  write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [63:0]           write_data,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [63:0]           read_data
);
    logic [ADDR_WIDTH-1:0] active_addr;
    logic [ADDR_WIDTH+6:0] active_addr_wide;
    logic [BANK_WIDTH-1:0] active_bank;
    logic [BANK_WIDTH-1:0] read_bank_q;
    logic [6:0] macro_addr;
    logic [63:0] bank_read_data [0:BANK_COUNT-1];

    assign active_addr = write_enable ? write_addr : read_addr;
    assign active_addr_wide = {{7{1'b0}}, active_addr};
    assign active_bank = active_addr_wide >> 7;
    assign macro_addr = active_addr_wide[6:0];
    assign read_data = bank_read_data[read_bank_q];

    always_ff @(posedge clk)
        read_bank_q <= active_bank;

    for (genvar bank = 0; bank < BANK_COUNT; bank++) begin : gen_sram_bank
        wire bank_selected = active_bank == BANK_WIDTH'(bank);

        fakeram7_128x64 macro (
            .rd_out(bank_read_data[bank]),
            .addr_in(macro_addr),
            .we_in(write_enable && bank_selected),
            .wd_in(write_data),
            .clk(clk),
            .ce_in(bank_selected)
        );
    end
endmodule
