module transformer_debug_counters (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear,
    input  logic        debug_enable,
    input  logic        debug_step,
    input  logic        freeze_counters,
    input  logic        accelerator_busy,
    input  logic        memory_stall,
    input  logic        input_stall,
    input  logic        output_stall,
    input  logic        op_start,
    input  logic        op_done,
    input  logic [7:0]  op_code,
    input  logic        error_event,
    input  logic [31:0] debug_data,
    output logic        pipeline_enable,
    output logic [63:0] total_cycles,
    output logic [63:0] busy_cycles,
    output logic [63:0] memory_stall_cycles,
    output logic [63:0] input_stall_cycles,
    output logic [63:0] output_stall_cycles,
    output logic [63:0] last_op_cycles,
    output logic [63:0] max_op_cycles,
    output logic [31:0] completed_ops,
    output logic [7:0]  active_op,
    output logic        error_latched,
    output logic        trace_valid,
    output logic [7:0]  trace_event,
    output logic [31:0] trace_data
);
    logic [63:0] op_started_at;

    assign pipeline_enable = !debug_enable || debug_step;

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            total_cycles <= '0;
            busy_cycles <= '0;
            memory_stall_cycles <= '0;
            input_stall_cycles <= '0;
            output_stall_cycles <= '0;
            last_op_cycles <= '0;
            max_op_cycles <= '0;
            completed_ops <= '0;
            active_op <= '0;
            op_started_at <= '0;
            error_latched <= 1'b0;
            trace_valid <= 1'b0;
            trace_event <= '0;
            trace_data <= '0;
        end else begin
            trace_valid <= 1'b0;
            if (!freeze_counters) begin
                total_cycles <= total_cycles + 1'b1;
                if (accelerator_busy) busy_cycles <= busy_cycles + 1'b1;
                if (memory_stall) memory_stall_cycles <= memory_stall_cycles + 1'b1;
                if (input_stall) input_stall_cycles <= input_stall_cycles + 1'b1;
                if (output_stall) output_stall_cycles <= output_stall_cycles + 1'b1;
            end
            if (op_start) begin
                active_op <= op_code;
                op_started_at <= total_cycles;
                if (debug_enable) begin
                    trace_valid <= 1'b1;
                    trace_event <= 8'h01;
                    trace_data <= {24'b0, op_code};
                end
            end
            if (op_done) begin
                last_op_cycles <= total_cycles - op_started_at;
                if (total_cycles - op_started_at > max_op_cycles)
                    max_op_cycles <= total_cycles - op_started_at;
                completed_ops <= completed_ops + 1'b1;
                if (debug_enable) begin
                    trace_valid <= 1'b1;
                    trace_event <= 8'h02;
                    trace_data <= total_cycles[31:0] - op_started_at[31:0];
                end
            end
            if (error_event) begin
                error_latched <= 1'b1;
                if (debug_enable) begin
                    trace_valid <= 1'b1;
                    trace_event <= 8'hff;
                    trace_data <= debug_data;
                end
            end else if (debug_enable && (memory_stall || input_stall || output_stall)) begin
                trace_valid <= 1'b1;
                trace_event <= {5'b0, output_stall, input_stall, memory_stall};
                trace_data <= debug_data;
            end
        end
    end
endmodule
