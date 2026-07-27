`timescale 1ns/1ps
module tb_int4_tile_group_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic group_begin_valid = 1'b0;
    logic group_begin_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic input_array = 1'b0;
    logic input_matrix = 1'b0;
    logic [2:0] input_row = '0;
    logic [7:0] input_mask = '0;
    logic [31:0] input_data = '0;
    logic group_commit_valid = 1'b0;
    logic group_commit_ready;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [2:0] output_row;
    logic [7:0] output_a_mask, output_b0_mask, output_b1_mask;
    logic [31:0] output_a_data, output_b0_data, output_b1_data;
    logic group_done;
    logic [1:0] committed_groups;

    integer output_count = 0;
    integer done_count = 0;
    integer expected_group;
    integer expected_index;
    logic output_started = 1'b0;
    integer stall_cycles = 0;
    logic stall_snapshot_valid = 1'b0;
    logic [2:0] stall_row;
    logic [95:0] stall_data;

    always #5 clk = ~clk;

    int4_tile_group_buffer dut (
        .clk, .rst_n,
        .group_begin_valid, .group_begin_ready,
        .input_valid, .input_ready,
        .input_array, .input_matrix, .input_row,
        .input_mask, .input_data,
        .group_commit_valid, .group_commit_ready,
        .output_valid, .output_ready,
        .output_row,
        .output_a_mask, .output_a_data,
        .output_b0_mask, .output_b0_data,
        .output_b1_mask, .output_b1_data,
        .group_done, .committed_groups
    );

    always @(posedge clk) begin
        if (rst_n) begin
            if (output_valid && !output_ready) begin
                stall_cycles <= stall_cycles + 1;
                if (!stall_snapshot_valid) begin
                    stall_snapshot_valid <= 1'b1;
                    stall_row <= output_row;
                    stall_data <= {output_b1_data, output_b0_data,
                                   output_a_data};
                end else if (output_row !== stall_row ||
                             {output_b1_data, output_b0_data,
                              output_a_data} !== stall_data) begin
                    $fatal(1, "SIMD output changed under backpressure");
                end
            end else if (output_valid && output_ready) begin
                stall_snapshot_valid <= 1'b0;
            end

            if (output_started && output_count < 16 &&
                output_count % 8 != 0 && !output_valid)
                $fatal(1, "bubble in committed tile-group stream at row %0d",
                       output_count);

            if (output_valid && output_ready) begin
                output_started <= 1'b1;
                expected_group = output_count / 8;
                expected_index = output_count % 8;
                if (output_a_data !== ((expected_group << 24) |
                                       (expected_index << 16) |
                                       (16'h4a00 + expected_index)))
                    $fatal(1, "A mismatch group=%0d row=%0d got=%08x",
                           expected_group, expected_index, output_a_data);
                if (output_b0_data !== ((expected_group << 24) |
                                        ((expected_index + 8) << 16) |
                                        (16'h4a00 + expected_index + 8)))
                    $fatal(1, "B0 mismatch group=%0d row=%0d got=%08x",
                           expected_group, expected_index, output_b0_data);
                if (output_b1_data !== ((expected_group << 24) |
                                        ((expected_index + 16) << 16) |
                                        (16'h4a00 + expected_index + 16)))
                    $fatal(1, "B1 mismatch group=%0d row=%0d got=%08x",
                           expected_group, expected_index, output_b1_data);
                if (output_a_mask !== 8'hff ||
                    output_b0_mask !== 8'hff ||
                    output_b1_mask !== 8'hff)
                    $fatal(1, "mask mismatch row=%0d", expected_index);
                if (output_row !== expected_index[2:0])
                    $fatal(1, "row mismatch expected=%0d got=%0d",
                           expected_index, output_row);
                output_count <= output_count + 1;
            end

            if (group_done)
                done_count <= done_count + 1;
        end
    end

    initial begin
        integer group_index;
        integer vector_index;

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;

        fork
            begin
                wait (output_count == 3);
                @(negedge clk);
                output_ready <= 1'b0;
                repeat (3) @(negedge clk);
                output_ready <= 1'b1;
            end
        join_none

        for (group_index = 0; group_index < 2; group_index = group_index + 1) begin
            for (vector_index = 0; vector_index < 24;
                 vector_index = vector_index + 1) begin
                @(negedge clk);
                group_begin_valid <= vector_index == 0;
                input_valid <= 1'b1;
                group_commit_valid <= vector_index == 23;
                input_matrix <= vector_index >= 8;
                input_array <= vector_index >= 16;
                input_row <= vector_index[2:0];
                input_mask <= 8'hff;
                input_data <= {group_index[7:0], vector_index[7:0],
                               16'h4a00 + vector_index[15:0]};
                @(posedge clk);
                if (!input_ready)
                    $fatal(1, "input stalled group=%0d vector=%0d",
                           group_index, vector_index);
                if (vector_index == 23 && !group_commit_ready)
                    $fatal(1, "commit stalled group=%0d", group_index);
            end
        end

        @(negedge clk);
        group_begin_valid <= 1'b0;
        input_valid <= 1'b0;
        group_commit_valid <= 1'b0;
        @(posedge clk);
        if (committed_groups != 2)
            $fatal(1, "second prefetched tile was not retained");
        @(negedge clk);
        output_ready <= 1'b1;

        wait (output_count == 16);
        @(posedge clk);
        if (done_count != 2)
            $fatal(1, "expected two group completions, got %0d", done_count);
        if (committed_groups != 0)
            $fatal(1, "buffer did not drain");
        if (stall_cycles < 3)
            $fatal(1, "backpressure interval was not exercised");

        $display("INT4 tile-group SIMD buffer golden trace passed: 48 vectors drained as 16 ordered rows, three vectors/cycle, stable backpressure");
        $finish;
    end

    initial begin
        repeat (200) @(posedge clk);
        $fatal(1, "tile-group buffer timeout");
    end
endmodule
