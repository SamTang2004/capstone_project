`timescale 1ns/1ps
module tb_gemm_controller_int4;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [15:0] M, N, K;
    logic busy, done, error;
    logic load_a_start, load_b_start, compute_start, write_c_start;
    logic load_a_done = 0, load_b_done = 0, compute_done = 0, write_c_done = 0;
    logic [15:0] m0, n0, k0;
    logic [3:0] m_len, k_len;
    logic [4:0] n_group_cols;
    logic [1:0] active_arrays;
    logic clear_accumulators, write_c_tiles;
    integer group_count = 0;
    integer compute_count = 0;
    integer write_count = 0;

    always #5 clk = ~clk;

    gemm_controller_int4 dut (
        .clk, .rst_n, .start, .M, .N, .K, .busy, .done, .error,
        .load_a_start, .load_a_done, .load_b_start, .load_b_done,
        .compute_start, .compute_done, .write_c_start, .write_c_done,
        .m0, .n0, .k0, .m_len, .k_len, .n_group_cols, .active_arrays,
        .clear_accumulators, .write_c_tiles
    );

    always_ff @(posedge clk) begin
        load_a_done <= load_a_start;
        load_b_done <= load_b_start;
        compute_done <= compute_start;
        write_c_done <= write_c_start;

        if (load_a_start) begin
            if (m0 == 0 && m_len != 8) $fatal(1, "bad first M extent");
            if (m0 == 8 && m_len != 5) $fatal(1, "bad tail M extent");
            if (k0 == 0 && k_len != 8) $fatal(1, "bad first K extent");
            if (k0 == 8 && k_len != 2) $fatal(1, "bad tail K extent");
            if (n0 == 0 && (n_group_cols != 16 || active_arrays != 2))
                $fatal(1, "bad full dual-array group");
            if (n0 == 16 && (n_group_cols != 3 || active_arrays != 1))
                $fatal(1, "bad N tail group");
            if (k0 == 0 && !clear_accumulators) $fatal(1, "clear missing");
            if (k0 != 0 && clear_accumulators) $fatal(1, "spurious clear");
            group_count <= group_count + 1;
        end
        if (compute_start) begin
            compute_count <= compute_count + 1;
            if ((k0 == 8) != write_c_tiles) $fatal(1, "write marker mismatch");
        end
        if (write_c_start) write_count <= write_count + 1;
    end

    initial begin
        M = 13;
        N = 19;
        K = 10;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait (done);
        if (error) $fatal(1, "unexpected controller error");
        if (group_count != 8) $fatal(1, "load group count %0d != 8", group_count);
        if (compute_count != 8) $fatal(1, "compute count %0d != 8", compute_count);
        if (write_count != 4) $fatal(1, "write count %0d != 4", write_count);
        $display("Dual INT4 controller golden trace passed: irregular 13x19x10 GEMM");
        $finish;
    end
endmodule
