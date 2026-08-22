// Reusable fabric multiplier for control/address setup. A single 8x16 byte
// products are reconstructed over registered stages, preserving DSP48s for MACs.
(* use_dsp = "no" *) module lut_mul32x16_pipeline (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        signed_mode,
    input  logic [31:0] lhs,
    input  logic [15:0] rhs,
    output logic        busy,
    output logic        done,
    output logic [47:0] product
);
    typedef enum logic [2:0] {
        IDLE, PARTIALS, ADD_PAIRS, COMBINE, APPLY_SIGN
    } state_t;
    state_t state;
    logic [31:0] lhs_abs_r;
    logic [15:0] rhs_r;
    logic negate_r;
    logic [23:0] partial0_r, partial1_r, partial2_r, partial3_r;
    logic [31:0] pair_low_r, pair_high_r;
    logic [47:0] unsigned_product_r;

    assign busy = state != IDLE;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            product <= '0;
            lhs_abs_r <= '0; rhs_r <= '0; negate_r <= 1'b0;
            partial0_r <= '0; partial1_r <= '0; partial2_r <= '0; partial3_r <= '0;
            pair_low_r <= '0; pair_high_r <= '0; unsigned_product_r <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    lhs_abs_r <= signed_mode && lhs[31] ? (~lhs + 1'b1) : lhs;
                    rhs_r <= rhs;
                    negate_r <= signed_mode && lhs[31];
                    state <= PARTIALS;
                end
                PARTIALS: begin
                    partial0_r <= lhs_abs_r[7:0] * rhs_r;
                    partial1_r <= lhs_abs_r[15:8] * rhs_r;
                    partial2_r <= lhs_abs_r[23:16] * rhs_r;
                    partial3_r <= lhs_abs_r[31:24] * rhs_r;
                    state <= ADD_PAIRS;
                end
                ADD_PAIRS: begin
                    pair_low_r <= {8'd0, partial0_r} + ({8'd0, partial1_r} << 8);
                    pair_high_r <= {8'd0, partial2_r} + ({8'd0, partial3_r} << 8);
                    state <= COMBINE;
                end
                COMBINE: begin
                    unsigned_product_r <= {16'd0, pair_low_r} + ({16'd0, pair_high_r} << 16);
                    state <= APPLY_SIGN;
                end
                APPLY_SIGN: begin
                    product <= negate_r ? (~unsigned_product_r + 1'b1) : unsigned_product_r;
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
