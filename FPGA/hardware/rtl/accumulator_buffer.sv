module accumulator_buffer #(
    parameter int TILE_SIZE = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    op_valid,
    output logic                    op_ready,
    input  logic                    op_clear,
    input  logic [7:0]              op_addr,
    input  logic signed [31:0]      op_data,
    output logic signed [31:0]      op_result,

    input  logic                    read_req,
    input  logic [7:0]              read_addr,
    output logic                    read_valid,
    output logic signed [31:0]      read_data
);

    localparam int ENTRY_COUNT = TILE_SIZE * TILE_SIZE;

    // Port A performs the serialized read-modify-write accumulation. Port B
    // is the registered tile-reader port consumed by tile_writer_ddr.
    (* ram_style = "block" *) logic signed [31:0] mem [0:ENTRY_COUNT-1];
    logic                    op_pending;
    logic                    pending_clear;
    logic [7:0]              pending_addr;
    logic signed [31:0]      pending_data;
    logic signed [31:0]      pending_old_value;
    logic signed [31:0]      pending_result;

    assign op_ready = !op_pending;
    assign pending_result = pending_clear ? pending_data : pending_old_value + pending_data;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            op_pending <= 1'b0;
            pending_clear <= 1'b0;
            pending_addr <= '0;
            pending_data <= '0;
            pending_old_value <= '0;
            op_result <= '0;
            read_valid <= 1'b0;
            read_data <= '0;
        end else begin
            read_valid <= read_req;
            if (read_req) begin
                read_data <= mem[read_addr];
            end

            if (op_pending) begin
                mem[pending_addr] <= pending_result;
                op_result <= pending_result;
                op_pending <= 1'b0;
            end

            if (op_valid && op_ready) begin
                pending_clear <= op_clear;
                pending_addr <= op_addr;
                pending_data <= op_data;
                pending_old_value <= mem[op_addr];
                op_pending <= 1'b1;
            end
        end
    end

endmodule
