// A local FIFO between a future AXI command adapter and tagged dispatch.
// Whole descriptors remain together; no tag lookup or task reordering.
module llm_command_queue #(
    parameter integer WIDTH = 128,
    parameter integer DEPTH = 4,
    parameter integer PTR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk, rst_n,
    input logic push_valid, output logic push_ready,
    input logic [WIDTH-1:0] push_data,
    output logic pop_valid, input logic pop_ready,
    output logic [WIDTH-1:0] pop_data,
    output logic [$clog2(DEPTH+1)-1:0] occupancy
);
    logic [WIDTH-1:0] entries [0:DEPTH-1];
    logic [PTR_WIDTH-1:0] rd, wr;
    logic push, pop;
    assign pop_valid = rst_n && occupancy != 0;
    assign pop = pop_valid && pop_ready;
    // A full queue can accept a replacement on the same edge it dispatches.
    assign push_ready = rst_n && (occupancy < DEPTH || pop);
    assign push = push_valid && push_ready;
    assign pop_data = pop_valid ? entries[rd] : '0;
    always_ff @(posedge clk) begin
        if (!rst_n) begin rd <= 0; wr <= 0; occupancy <= 0; end
        else begin
            if (push) begin
                entries[wr] <= push_data;
                wr <= wr == DEPTH-1 ? '0 : wr + 1'b1;
            end
            if (pop) rd <= rd == DEPTH-1 ? '0 : rd + 1'b1;
            case ({push,pop})
                2'b10: occupancy <= occupancy + 1'b1;
                2'b01: occupancy <= occupancy - 1'b1;
                default: occupancy <= occupancy;
            endcase
        end
    end
endmodule
