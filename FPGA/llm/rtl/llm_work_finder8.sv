// Eight-entry stable filter/compactor. Payload travels with its original value.
// X discards bits [X-1:0]; keep iff any bit at position X or above is set.
// SIGNED_MAGNITUDE=1 tests abs(two's-complement value), without changing value.
module llm_work_finder8 #(
    parameter integer WIDTH = 8,
    parameter integer PAYLOAD_WIDTH = 32,
    parameter integer SIGNED_MAGNITUDE = 1,
    parameter integer SHIFT_WIDTH = $clog2(WIDTH + 1)
) (
    input logic clk, rst_n,
    input logic in_valid,
    output logic in_ready,
    input logic [7:0] in_mask,
    input logic [SHIFT_WIDTH-1:0] in_shift,
    input logic [8*WIDTH-1:0] in_values,
    input logic [8*PAYLOAD_WIDTH-1:0] in_payload,
    output logic out_valid,
    input logic out_ready,
    output logic [3:0] out_count,
    output logic [7:0] out_mask,
    output logic [7:0] out_keep_mask,
    output logic [8*WIDTH-1:0] out_values,
    output logic [8*PAYLOAD_WIDTH-1:0] out_payload,
    output logic [23:0] out_source_lanes
);
    logic valid_q;
    logic [3:0] count_d;
    logic [7:0] mask_d, keep_d;
    logic [8*WIDTH-1:0] values_d;
    logic [8*PAYLOAD_WIDTH-1:0] payload_d;
    logic [23:0] sources_d;
    logic [WIDTH-1:0] value_bits, magnitude;
    logic [3:0] ranks [0:8];
    integer lane, slot;

    assign out_valid = rst_n && valid_q;
    assign in_ready = rst_n && (!valid_q || out_ready);

    always_comb begin
        count_d = '0;
        mask_d = '0;
        keep_d = '0;
        values_d = '0;
        payload_d = '0;
        sources_d = '0;
        value_bits = '0;
        magnitude = '0;
        ranks[0] = 0;
        for (lane = 0; lane < 8; lane = lane + 1) begin
            value_bits = in_values[lane*WIDTH +: WIDTH];
            // Unsigned magnitude can represent abs(-128)=128 at WIDTH=8.
            magnitude = (SIGNED_MAGNITUDE && value_bits[WIDTH-1])
                      ? (~value_bits + 1'b1) : value_bits;
            if (in_mask[lane] && (in_shift < WIDTH) &&
                (|(magnitude >> in_shift))) begin
                keep_d[lane] = 1'b1;
            end
            ranks[lane+1] = ranks[lane] + {3'b0, keep_d[lane]};
        end
        count_d = ranks[8];
        // Fixed slices and mutually exclusive selectors avoid wide variable
        // part-select writes (which otherwise infer expensive barrel muxes).
        for (slot = 0; slot < 8; slot = slot + 1) begin
            mask_d[slot] = slot < count_d;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                if (keep_d[lane] && ranks[lane] == slot) begin
                    values_d[slot*WIDTH +: WIDTH] = values_d[slot*WIDTH +: WIDTH] |
                        in_values[lane*WIDTH +: WIDTH];
                    payload_d[slot*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] = payload_d[slot*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] |
                        in_payload[lane*PAYLOAD_WIDTH +: PAYLOAD_WIDTH];
                    sources_d[slot*3 +: 3] = sources_d[slot*3 +: 3] | 3'(lane);
                end
            end
        end
    end

    // One elastic output register; can consume and replace on the same edge.
    // A zero-survivor input still emits a valid, count=0 completion packet.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_q <= 1'b0;
            out_count <= '0;
            out_mask <= '0;
            out_keep_mask <= '0;
            out_values <= '0;
            out_payload <= '0;
            out_source_lanes <= '0;
        end else if (in_ready) begin
            valid_q <= in_valid;
            if (in_valid) begin
                out_count <= count_d;
                out_mask <= mask_d;
                out_keep_mask <= keep_d;
                out_values <= values_d;
                out_payload <= payload_d;
                out_source_lanes <= sources_d;
            end
        end
    end
endmodule
