module simple_dual_port_bram #(
    parameter int DATA_WIDTH = 64,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input  logic                  clk,
    input  logic                  write_enable,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [DATA_WIDTH-1:0] write_data,
    input  logic                  read_enable,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [DATA_WIDTH-1:0] read_data
);
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (write_enable) memory[write_addr] <= write_data;
        if (read_enable) read_data <= memory[read_addr];
    end
endmodule
