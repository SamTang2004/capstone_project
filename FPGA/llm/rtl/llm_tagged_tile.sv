// LLM overlay v0 compute boundary. No DeiT graph ABI or DDR addresses.
// One outstanding tile; completion must be consumed before the next command.
module llm_tagged_tile (
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [15:0] cmd_batch,
    input logic [31:0] cmd_task,
    input logic cmd_w4,
    input logic [15:0] cmd_k,
    input logic [3:0] cmd_rows, cmd_cols,
    input logic operand_valid, output logic operand_ready,
    input logic [63:0] operand_a, operand_b,
    output logic completion_valid, input logic completion_ready,
    output logic [15:0] completion_batch,
    output logic [31:0] completion_task,
    output logic [1:0] completion_status,
    output logic [31:0] execution_cycles, input_wait_cycles,
    output logic [2047:0] completion_accumulators
);
    typedef enum logic [1:0] {IDLE, RUN, COMPLETE} state_t;
    state_t state;
    logic legal, launch, core_ready, core_valid, core_error, core_busy;
    logic [2047:0] core_accumulators;
    logic [31:0] elapsed, waits;
    assign legal = cmd_k != 0 && cmd_rows >= 1 && cmd_rows <= 8 &&
                   cmd_cols >= 1 && cmd_cols <= 8;
    assign cmd_ready = rst_n && state == IDLE;
    assign launch = cmd_valid && cmd_ready && legal;
    assign operand_ready = state == RUN && core_ready;
    assign completion_valid = state == COMPLETE;

    gemm_8x8_tile_core core (
        .clk, .rst_n, .start(launch), .weight_int4(cmd_w4),
        .k_steps(cmd_k), .valid_rows(cmd_rows), .valid_cols(cmd_cols),
        .input_valid(operand_valid && state == RUN), .input_ready(core_ready),
        .activation_vector(operand_a), .packed_weight_vector(operand_b),
        .busy(core_busy), .result_valid(core_valid), .result_ready(state == RUN),
        .error(core_error), .accumulators(core_accumulators)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            completion_batch <= 0; completion_task <= 0;
            completion_status <= 0; completion_accumulators <= 0;
            elapsed <= 0; waits <= 0;
            execution_cycles <= 0; input_wait_cycles <= 0;
        end else case (state)
            IDLE: if (cmd_valid && cmd_ready) begin
                completion_batch <= cmd_batch; completion_task <= cmd_task;
                completion_status <= legal ? 2'd0 : 2'd1;
                elapsed <= 0; waits <= 0;
                execution_cycles <= 0; input_wait_cycles <= 0;
                completion_accumulators <= 0;
                state <= legal ? RUN : COMPLETE;
            end
            RUN: begin
                elapsed <= elapsed + 1'b1;
                if (operand_ready && !operand_valid) waits <= waits + 1'b1;
                if (core_valid || core_error) begin
                    completion_status <= core_error ? 2'd2 : 2'd0;
                    completion_accumulators <= core_error ? '0 : core_accumulators;
                    execution_cycles <= elapsed + 1'b1;
                    input_wait_cycles <= waits;
                    state <= COMPLETE;
                end
            end
            COMPLETE: if (completion_ready) state <= IDLE;
            default: state <= IDLE;
        endcase
    end
endmodule
