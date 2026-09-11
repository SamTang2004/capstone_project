module bcuda_asap7_sram_1rw_8 #(
    parameter int DEPTH = 512,
    parameter int ADDR_WIDTH = DEPTH <= 1 ? 1 : $clog2(DEPTH),
    localparam int MACRO_DEPTH = 512,
    localparam int BANK_COUNT = (DEPTH + MACRO_DEPTH - 1) / MACRO_DEPTH,
    localparam int BANK_WIDTH = BANK_COUNT <= 1 ? 1 : $clog2(BANK_COUNT)
) (
    input  logic                  clk,
    input  logic                  write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [7:0]            write_data,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [7:0]            read_data
);
    logic [ADDR_WIDTH-1:0] active_addr;
    logic [ADDR_WIDTH+8:0] active_addr_wide;
    logic [BANK_WIDTH-1:0] active_bank, read_bank_q;
    logic [8:0] macro_addr;
    logic [7:0] bank_read_data [0:BANK_COUNT-1];

    assign active_addr = write_enable ? write_addr : read_addr;
    assign active_addr_wide = {{9{1'b0}}, active_addr};
    assign active_bank = active_addr_wide >> 9;
    assign macro_addr = active_addr_wide[8:0];
    assign read_data = bank_read_data[read_bank_q];

    always_ff @(posedge clk)
        read_bank_q <= active_bank;

    for (genvar bank = 0; bank < BANK_COUNT; bank++) begin : gen_sram_bank
        wire bank_selected = active_bank == BANK_WIDTH'(bank);
        fakeram_512x8 macro (
            .rd_out(bank_read_data[bank]), .addr_in(macro_addr),
            .we_in(write_enable && bank_selected), .wd_in(write_data),
            .clk(clk), .ce_in(bank_selected)
        );
    end
endmodule
