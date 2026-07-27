`timescale 1ns/1ps
module tb_int4_conv_weight_tile_cache;
    logic clk = 0;
    logic rst_n = 0;
    logic clear = 0;
    logic [2:0] query_index;
    logic query_hit;
    logic capture_start = 0;
    logic [2:0] capture_index;
    logic [1:0] capture_active_arrays;
    logic capture_valid = 0;
    logic capture_ready;
    logic capture_array, capture_matrix;
    logic [2:0] capture_row;
    logic [7:0] capture_mask;
    logic [31:0] capture_data;
    logic capture_finish = 0;
    logic replay_start = 0;
    logic [2:0] replay_index;
    logic [1:0] replay_active_arrays;
    logic replay_busy, replay_done, replay_error;
    logic replay_valid, replay_ready = 1;
    logic replay_array, replay_matrix;
    logic [2:0] replay_row;
    logic [7:0] replay_mask;
    logic [31:0] replay_data;
    integer replay_count = 0;
    integer stall_count = 0;

    always #5 clk = ~clk;

    int4_conv_weight_tile_cache #(
        .TILE_COUNT(8), .INDEX_W(3)
    ) dut (
        .clk, .rst_n, .clear, .query_index, .query_hit,
        .capture_start, .capture_index,
        .capture_valid, .capture_ready, .capture_array, .capture_matrix,
        .capture_row, .capture_mask, .capture_data, .capture_finish,
        .replay_start, .replay_index, .replay_active_arrays,
        .replay_busy, .replay_done, .replay_error,
        .replay_valid, .replay_ready, .replay_array, .replay_matrix,
        .replay_row, .replay_mask, .replay_data
    );

    always_ff @(posedge clk) begin
        if (replay_valid && replay_ready) begin
            if (!replay_matrix)
                $fatal(1, "cache replayed a non-weight vector");
            if (replay_array !== (replay_count >= 8))
                $fatal(1, "array mismatch at replay row %0d", replay_count);
            if (replay_row !== replay_count[2:0])
                $fatal(1, "row mismatch at replay row %0d", replay_count);
            if (replay_mask !== (replay_count < 8 ? 8'hff : 8'h0f))
                $fatal(1, "mask mismatch at replay row %0d", replay_count);
            if (replay_data !== (32'h4100_0000 + replay_count))
                $fatal(1, "data mismatch at replay row %0d", replay_count);
            replay_count <= replay_count + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        query_index = 3;
        capture_index = 3;
        capture_active_arrays = 2;
        replay_index = 3;
        replay_active_arrays = 2;
        @(posedge clk);
        if (query_hit)
            $fatal(1, "empty cache reported a hit");

        capture_start = 1;
        @(posedge clk);
        #1;
        capture_start = 0;
        for (int row = 0; row < 16; row++) begin
            @(negedge clk);
            capture_valid = 1;
            capture_matrix = 1;
            capture_array = row >= 8;
            capture_row = row[2:0];
            capture_mask = row < 8 ? 8'hff : 8'h0f;
            capture_data = 32'h4100_0000 + row;
        end
        @(negedge clk);
        capture_valid = 0;
        capture_finish = 1;
        @(posedge clk);
        #1;
        capture_finish = 0;
        #1;
        if (!query_hit)
            $fatal(1, "completed tile did not become valid");

        replay_start = 1;
        @(posedge clk);
        #1;
        replay_start = 0;
        while (!replay_done) begin
            replay_ready = stall_count[1:0] != 2'b01;
            stall_count = stall_count + 1;
            @(posedge clk);
        end
        replay_ready = 1;
        if (replay_error || replay_count != 16)
            $fatal(1, "replay failed: error=%0d rows=%0d",
                   replay_error, replay_count);

        clear = 1;
        @(posedge clk);
        clear = 0;
        #1;
        if (query_hit)
            $fatal(1, "clear did not invalidate cache");
        $display("INT4 persistent Conv weight cache passed: 16 BRAM rows replayed with backpressure");
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "timeout");
    end
endmodule
