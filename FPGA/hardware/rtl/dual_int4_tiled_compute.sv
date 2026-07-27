`timescale 1ns/1ps
module dual_int4_tiled_compute (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [15:0]        M,
    input  logic [15:0]        N,
    input  logic [15:0]        K,
    output logic               busy,
    output logic               done,
    output logic               error,
    output logic               compute_active,
    output logic               load_wait_active,

    output logic               load_a_request,
    output logic               load_b_request,
    input  logic               external_load_done,
    input  logic               external_load_preloaded,
    input  logic               external_load_valid,
    output logic               external_load_ready,
    input  logic               external_load_array,
    input  logic               external_load_matrix,
    input  logic [2:0]         external_load_row,
    input  logic [7:0]         external_load_mask,
    input  logic [31:0]        external_load_data,
    input  logic               external_group_valid,
    output logic               external_group_ready,
    input  logic [2:0]         external_group_row,
    input  logic [7:0]         external_group_a_mask,
    input  logic [31:0]        external_group_a_data,
    input  logic [7:0]         external_group_b0_mask,
    input  logic [31:0]        external_group_b0_data,
    input  logic [7:0]         external_group_b1_mask,
    input  logic [31:0]        external_group_b1_data,

    input  logic               seed_valid,
    output logic               seed_ready,
    input  logic               seed_array,
    input  logic [5:0]         seed_addr,
    input  logic signed [31:0] seed_data,
    input  logic               preserve_accumulator_seed,

    output logic [15:0]        m0,
    output logic [15:0]        n0,
    output logic [15:0]        k0,
    output logic [3:0]         m_len,
    output logic [3:0]         k_len,
    output logic [4:0]         n_group_cols,
    output logic [1:0]         active_arrays,

    output logic               output_valid,
    input  logic               output_ready,
    output logic               output_array,
    output logic [5:0]         output_addr,
    output logic signed [31:0] output_data
);

    typedef enum logic [3:0] {
        OP_IDLE,
        OP_ZERO_A,
        OP_WAIT_A,
        OP_ZERO_B0,
        OP_ZERO_B1,
        OP_WAIT_B,
        OP_COMPUTE_WAIT,
        OP_ACCUMULATE,
        OP_ACCUMULATE_DRAIN,
        OP_WRITE_REQ,
        OP_WRITE_WAIT
    } op_state_t;

    op_state_t op_state;
    logic ctl_load_a_start;
    logic ctl_load_a_done;
    logic ctl_load_b_start;
    logic ctl_load_b_done;
    logic ctl_compute_start;
    logic ctl_compute_done;
    logic ctl_write_start;
    logic ctl_write_done;
    logic clear_accumulators;
    logic write_c_tiles;
    logic core_start;
    logic core_busy;
    logic core_done;
    logic core_load_valid;
    logic core_load_ready;
    logic core_group_ready;
    logic core_load_array;
    logic core_load_matrix;
    logic [2:0] core_load_row;
    logic [7:0] core_load_mask;
    logic [31:0] core_load_data;
    logic signed [255:0] core_result0_row;
    logic signed [255:0] core_result1_row;
    logic [2:0] zero_idx;
    logic [2:0] accum_row;
    logic accum_row_ready;
    logic accum_row_commit;
    logic [2:0] accum_row_commit_addr;
    logic accum_read_valid;
    logic signed [31:0] accum_read_data;
    logic write_array;
    logic [2:0] write_row;
    logic [2:0] write_col;
    logic [3:0] write_cols;
    logic output_holding_valid;
    logic output_holding_array;
    logic [5:0] output_holding_addr;
    logic signed [31:0] output_holding_data;

    assign load_a_request = op_state == OP_ZERO_A ||
                            op_state == OP_ZERO_B0 ||
                            op_state == OP_ZERO_B1 ||
                            op_state == OP_WAIT_A;
    assign compute_active = core_busy;
    assign load_wait_active = op_state == OP_WAIT_A;
    assign load_b_request = ctl_load_b_start;
    assign core_start = ctl_compute_start;
    assign core_load_valid = (op_state == OP_ZERO_A ||
                              op_state == OP_ZERO_B0 ||
                              op_state == OP_ZERO_B1) ?
                             1'b1 : external_load_valid;
    assign core_load_array = (op_state == OP_ZERO_B1) ? 1'b1 :
                             (op_state == OP_ZERO_A || op_state == OP_ZERO_B0) ? 1'b0 :
                             external_load_array;
    assign core_load_matrix = (op_state == OP_ZERO_B0 || op_state == OP_ZERO_B1) ?
                              1'b1 :
                              (op_state == OP_ZERO_A) ? 1'b0 : external_load_matrix;
    assign core_load_row = (op_state == OP_ZERO_A ||
                            op_state == OP_ZERO_B0 ||
                            op_state == OP_ZERO_B1) ? zero_idx : external_load_row;
    assign core_load_mask = (op_state == OP_ZERO_A ||
                             op_state == OP_ZERO_B0 ||
                             op_state == OP_ZERO_B1) ? 8'hff : external_load_mask;
    assign core_load_data = (op_state == OP_ZERO_A ||
                             op_state == OP_ZERO_B0 ||
                             op_state == OP_ZERO_B1) ? 32'd0 : external_load_data;
    assign external_load_ready = (op_state == OP_WAIT_A ||
                                  op_state == OP_ACCUMULATE ||
                                  op_state == OP_ACCUMULATE_DRAIN) &&
                                 core_load_ready;
    assign external_group_ready = (op_state == OP_WAIT_A ||
                                   op_state == OP_ACCUMULATE ||
                                   op_state == OP_ACCUMULATE_DRAIN) &&
                                  core_group_ready;

    assign output_valid = output_holding_valid;
    assign output_array = output_holding_array;
    assign output_addr = output_holding_addr;
    assign output_data = output_holding_data;

    gemm_controller_int4 controller (
        .clk, .rst_n, .start, .M, .N, .K, .busy, .done, .error,
        .load_a_start(ctl_load_a_start), .load_a_done(ctl_load_a_done),
        .load_b_start(ctl_load_b_start), .load_b_done(ctl_load_b_done),
        .compute_start(ctl_compute_start), .compute_done(ctl_compute_done),
        .write_c_start(ctl_write_start), .write_c_done(ctl_write_done),
        .m0, .n0, .k0, .m_len, .k_len, .n_group_cols, .active_arrays,
        .clear_accumulators, .write_c_tiles
    );

    gemm_dual_8x8_int4_core core (
        .clk, .rst_n,
        .load_valid(1'b0), .load_array(1'b0), .load_matrix(1'b0),
        .load_addr(6'd0), .load_data(4'sd0),
        .load_vector_valid(core_load_valid), .load_vector_ready(core_load_ready),
        .load_vector_array(core_load_array), .load_vector_matrix(core_load_matrix),
        .load_vector_row(core_load_row), .load_vector_mask(core_load_mask),
        .load_vector_data(core_load_data),
        .load_group_valid(external_group_valid &&
                          (op_state == OP_WAIT_A ||
                           op_state == OP_ACCUMULATE ||
                           op_state == OP_ACCUMULATE_DRAIN)),
        .load_group_ready(core_group_ready),
        .load_group_row(external_group_row),
        .load_group_a_mask(external_group_a_mask),
        .load_group_a_data(external_group_a_data),
        .load_group_b0_mask(external_group_b0_mask),
        .load_group_b0_data(external_group_b0_data),
        .load_group_b1_mask(external_group_b1_mask),
        .load_group_b1_data(external_group_b1_data),
        .start(core_start),
        .share_mode(2'b01), .busy(core_busy), .done(core_done),
        .result_array(1'b0), .result_addr(6'd0), .result_data(),
        .result_row(accum_row),
        .result0_row(core_result0_row), .result1_row(core_result1_row)
    );

    parallel_accumulator_buffer accumulator (
        .clk, .rst_n,
        .row_valid(op_state == OP_ACCUMULATE),
        .row_ready(accum_row_ready),
        .row_clear(clear_accumulators && !preserve_accumulator_seed),
        .row_active_arrays(active_arrays),
        .row_addr(accum_row),
        .row_data0(core_result0_row), .row_data1(core_result1_row),
        .row_commit(accum_row_commit),
        .row_commit_addr(accum_row_commit_addr),
        .seed_valid, .seed_ready, .seed_array, .seed_addr, .seed_data,
        .read_req(op_state == OP_WRITE_REQ),
        .read_array(write_array),
        .read_addr({write_row, write_col}),
        .read_valid(accum_read_valid), .read_data(accum_read_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            op_state <= OP_IDLE;
            ctl_load_a_done <= 1'b0;
            ctl_load_b_done <= 1'b0;
            ctl_compute_done <= 1'b0;
            ctl_write_done <= 1'b0;
            zero_idx <= '0;
            accum_row <= '0;
            write_array <= 1'b0;
            write_row <= '0;
            write_col <= '0;
            write_cols <= '0;
            output_holding_valid <= 1'b0;
            output_holding_array <= 1'b0;
            output_holding_addr <= '0;
            output_holding_data <= '0;
        end else begin
            ctl_load_a_done <= 1'b0;
            ctl_load_b_done <= 1'b0;
            ctl_compute_done <= 1'b0;
            ctl_write_done <= 1'b0;

            unique case (op_state)
                OP_IDLE: begin
                    if (ctl_load_a_start) begin
                        zero_idx <= '0;
                        op_state <= external_load_preloaded ?
                                    OP_WAIT_A : OP_ZERO_A;
                    end else if (ctl_load_b_start) begin
                        // A, B0, and B1 arrive as one committed tile group.
                        ctl_load_b_done <= 1'b1;
                    end else if (ctl_compute_start) begin
                        op_state <= OP_COMPUTE_WAIT;
                    end else if (ctl_write_start) begin
                        write_array <= 1'b0;
                        write_row <= '0;
                        write_col <= '0;
                        write_cols <= (n_group_cols > 8) ? 4'd8 : n_group_cols[3:0];
                        op_state <= OP_WRITE_REQ;
                    end
                end

                OP_ZERO_A: begin
                    if (core_load_ready && zero_idx == 3'd7) begin
                        zero_idx <= '0;
                        op_state <= OP_ZERO_B0;
                    end else if (core_load_ready) zero_idx <= zero_idx + 1'b1;
                end

                OP_WAIT_A: if (external_load_done) begin
                    ctl_load_a_done <= 1'b1;
                    op_state <= OP_IDLE;
                end

                OP_ZERO_B0: begin
                    if (core_load_ready && zero_idx == 3'd7) begin
                        zero_idx <= '0;
                        op_state <= OP_ZERO_B1;
                    end else if (core_load_ready) zero_idx <= zero_idx + 1'b1;
                end

                OP_ZERO_B1: begin
                    if (core_load_ready && zero_idx == 3'd7) begin
                        zero_idx <= '0;
                        op_state <= OP_WAIT_A;
                    end else if (core_load_ready) zero_idx <= zero_idx + 1'b1;
                end

                OP_WAIT_B: op_state <= OP_IDLE;

                OP_COMPUTE_WAIT: if (core_done) begin
                    accum_row <= '0;
                    op_state <= OP_ACCUMULATE;
                end

                OP_ACCUMULATE: begin
                    if (accum_row_ready) begin
                        if (accum_row == 7)
                            op_state <= OP_ACCUMULATE_DRAIN;
                        else
                            accum_row <= accum_row + 1'b1;
                    end
                end

                OP_ACCUMULATE_DRAIN: begin
                    if (accum_row_commit && accum_row_commit_addr == 7) begin
                        ctl_compute_done <= 1'b1;
                        op_state <= OP_IDLE;
                    end
                end

                OP_WRITE_REQ: op_state <= OP_WRITE_WAIT;

                OP_WRITE_WAIT: begin
                    if (accum_read_valid && !output_holding_valid) begin
                        output_holding_valid <= 1'b1;
                        output_holding_array <= write_array;
                        output_holding_addr <= {write_row, write_col};
                        output_holding_data <= accum_read_data;
                    end
                    if (output_holding_valid && output_ready) begin
                        output_holding_valid <= 1'b0;
                        if (write_col + 1'b1 < write_cols) begin
                            write_col <= write_col + 1'b1;
                            op_state <= OP_WRITE_REQ;
                        end else if (write_row + 1'b1 < m_len) begin
                            write_col <= '0;
                            write_row <= write_row + 1'b1;
                            op_state <= OP_WRITE_REQ;
                        end else if (!write_array && active_arrays == 2) begin
                            write_array <= 1'b1;
                            write_row <= '0;
                            write_col <= '0;
                            write_cols <= n_group_cols - 8;
                            op_state <= OP_WRITE_REQ;
                        end else begin
                            ctl_write_done <= 1'b1;
                            op_state <= OP_IDLE;
                        end
                    end
                end

                default: op_state <= OP_IDLE;
            endcase
        end
    end
endmodule
