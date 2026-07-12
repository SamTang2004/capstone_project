module tb_gemm_controller;
    logic clk;
    logic rst_n;
    logic start;
    logic [15:0] M;
    logic [15:0] N;
    logic [15:0] K;
    logic [4:0] tile_size;
    logic [7:0] array_count;
    logic busy;
    logic done;
    logic error;
    logic load_a_start;
    logic load_a_done;
    logic load_b_start;
    logic load_b_done;
    logic compute_start;
    logic compute_done;
    logic write_c_start;
    logic write_c_done;
    logic [15:0] m_tile;
    logic [15:0] n_group_tile;
    logic [15:0] k_tile;
    logic [15:0] m0;
    logic [15:0] n0;
    logic [15:0] k0;
    logic [4:0] m_len;
    logic [15:0] n_group_cols;
    logic [4:0] k_len;
    logic [7:0] active_arrays;
    logic clear_accumulators;
    logic write_c_tiles;

    int load_a_count;
    int load_b_count;
    int compute_count;
    int write_count;
    int edge_seen;

    gemm_controller dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .M(M),
        .N(N),
        .K(K),
        .tile_size(tile_size),
        .array_count(array_count),
        .busy(busy),
        .done(done),
        .error(error),
        .load_a_start(load_a_start),
        .load_a_done(load_a_done),
        .load_b_start(load_b_start),
        .load_b_done(load_b_done),
        .compute_start(compute_start),
        .compute_done(compute_done),
        .write_c_start(write_c_start),
        .write_c_done(write_c_done),
        .m_tile(m_tile),
        .n_group_tile(n_group_tile),
        .k_tile(k_tile),
        .m0(m0),
        .n0(n0),
        .k0(k0),
        .m_len(m_len),
        .n_group_cols(n_group_cols),
        .k_len(k_len),
        .active_arrays(active_arrays),
        .clear_accumulators(clear_accumulators),
        .write_c_tiles(write_c_tiles)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            load_a_done <= 1'b0;
            load_b_done <= 1'b0;
            compute_done <= 1'b0;
            write_c_done <= 1'b0;
        end else begin
            load_a_done <= load_a_start;
            load_b_done <= load_b_start;
            compute_done <= compute_start;
            write_c_done <= write_c_start;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            load_a_count <= 0;
            load_b_count <= 0;
            compute_count <= 0;
            write_count <= 0;
            edge_seen <= 0;
        end else begin
            if (load_a_start) load_a_count <= load_a_count + 1;
            if (load_b_start) load_b_count <= load_b_count + 1;
            if (compute_start) begin
                compute_count <= compute_count + 1;
                if (m_tile == 0 && n_group_tile == 0 && k_tile == 0) begin
                    if (m0 != 0 || n0 != 0 || k0 != 0 || m_len != 16 || n_group_cols != 32 || k_len != 16 || active_arrays != 2 || !clear_accumulators) begin
                        $fatal(1, "first tile metadata mismatch");
                    end
                end
                if (m_tile == 2 && n_group_tile == 2 && k_tile == 1) begin
                    edge_seen <= 1;
                    if (m0 != 32 || n0 != 32 || k0 != 16 || m_len != 3 || n_group_cols != 9 || k_len != 11 || active_arrays != 1 || !write_c_tiles) begin
                        $fatal(1, "ragged edge metadata mismatch");
                    end
                end
            end
            if (write_c_start) write_count <= write_count + 1;
        end
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        M = 16'd35;
        N = 16'd41;
        K = 16'd27;
        tile_size = 5'd16;
        array_count = 8'd2;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (500) begin
            @(posedge clk);
            if (done) begin
                if (error) begin
                    $fatal(1, "controller reported error");
                end
                if (load_a_count != 12 || load_b_count != 12 || compute_count != 12 || write_count != 6 || !edge_seen) begin
                    $fatal(1, "unexpected schedule counts: load_a=%0d load_b=%0d compute=%0d write=%0d edge=%0d",
                        load_a_count, load_b_count, compute_count, write_count, edge_seen);
                end
                $display("tb_gemm_controller golden trace passed");
                $finish;
            end
        end

        $fatal(1, "timeout waiting for controller completion");
    end

endmodule
