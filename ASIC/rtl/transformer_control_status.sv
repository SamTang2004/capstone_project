module transformer_control_status (
    input  logic          clk, rst_n,
    input  logic [31:0]   debug_ctrl_word,
    input  logic          accelerator_busy, memory_stall, input_stall, output_stall,
    input  logic          op_start, op_done,
    input  logic [7:0]    op_code,
    input  logic          error_event,
    input  logic [31:0]   debug_data,
    output logic          pipeline_enable,
    output logic [32*18-1:0] debug_regs_flat
);
    logic [63:0] total_cycles, busy_cycles, memory_stall_cycles, input_stall_cycles;
    logic [63:0] output_stall_cycles, last_op_cycles, max_op_cycles;
    logic [31:0] completed_ops, trace_data;
    logic [7:0] active_op, trace_event;
    logic error_latched, trace_valid;
    logic accelerator_busy_q, memory_stall_q, input_stall_q, output_stall_q;
    logic op_start_q, op_done_q, error_event_q;
    logic [7:0] op_code_q;
    logic [31:0] debug_data_q;

    // Keep the optional debug/trace machinery off the accelerator's shared-DDR
    // handshake paths.  Events are delayed by one cycle, while their widths and
    // ordering are preserved, so cycle totals and per-operation deltas remain
    // unchanged.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            accelerator_busy_q <= 1'b0;
            memory_stall_q <= 1'b0;
            input_stall_q <= 1'b0;
            output_stall_q <= 1'b0;
            op_start_q <= 1'b0;
            op_done_q <= 1'b0;
            op_code_q <= '0;
            error_event_q <= 1'b0;
            debug_data_q <= '0;
        end else begin
            accelerator_busy_q <= accelerator_busy;
            memory_stall_q <= memory_stall;
            input_stall_q <= input_stall;
            output_stall_q <= output_stall;
            op_start_q <= op_start;
            op_done_q <= op_done;
            op_code_q <= op_code;
            error_event_q <= error_event;
            debug_data_q <= debug_data;
        end
    end

    transformer_debug_counters counters (
        .clk, .rst_n, .clear(debug_ctrl_word[3]), .debug_enable(debug_ctrl_word[0]),
        .debug_step(debug_ctrl_word[1]), .freeze_counters(debug_ctrl_word[2]),
        .accelerator_busy(accelerator_busy_q), .memory_stall(memory_stall_q),
        .input_stall(input_stall_q), .output_stall(output_stall_q),
        .op_start(op_start_q), .op_done(op_done_q), .op_code(op_code_q),
        .error_event(error_event_q), .debug_data(debug_data_q), .pipeline_enable,
        .total_cycles, .busy_cycles, .memory_stall_cycles, .input_stall_cycles,
        .output_stall_cycles, .last_op_cycles, .max_op_cycles, .completed_ops,
        .active_op, .error_latched, .trace_valid, .trace_event, .trace_data
    );

    always_comb begin
        debug_regs_flat = '0;
        debug_regs_flat[0*32 +: 64] = total_cycles;
        debug_regs_flat[2*32 +: 64] = busy_cycles;
        debug_regs_flat[4*32 +: 64] = memory_stall_cycles;
        debug_regs_flat[6*32 +: 64] = input_stall_cycles;
        debug_regs_flat[8*32 +: 64] = output_stall_cycles;
        debug_regs_flat[10*32 +: 64] = last_op_cycles;
        debug_regs_flat[12*32 +: 64] = max_op_cycles;
        debug_regs_flat[14*32 +: 32] = completed_ops;
        debug_regs_flat[15*32 +: 32] = {22'b0, trace_valid, error_latched, active_op};
        debug_regs_flat[16*32 +: 32] = {24'b0, trace_event};
        debug_regs_flat[17*32 +: 32] = trace_data;
    end
endmodule
