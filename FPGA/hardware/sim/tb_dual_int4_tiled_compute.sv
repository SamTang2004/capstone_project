`timescale 1ns/1ps
module tb_dual_int4_tiled_compute;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [15:0] M = 13, N = 19, K = 10;
    logic busy, done, error;
    logic load_a_request, load_b_request;
    logic external_load_done = 0;
    logic external_load_valid = 0;
    logic external_load_ready;
    logic external_load_array = 0;
    logic external_load_matrix = 0;
    logic [2:0] external_load_row = 0;
    logic [7:0] external_load_mask = 0;
    logic [31:0] external_load_data = 0;
    logic [15:0] m0, n0, k0;
    logic [3:0] m_len, k_len;
    logic [4:0] n_group_cols;
    logic [1:0] active_arrays;
    logic output_valid;
    logic output_ready = 1;
    logic output_array;
    logic [5:0] output_addr;
    logic signed [31:0] output_data;
    integer outputs = 0;
    integer row, col, array_id;
    integer expected, global_row, global_col;

    always #5 clk = ~clk;

    function automatic integer aval(input integer r, input integer k);
        aval = ((r * 3 + k * 5 + 4) & 15) - 8;
    endfunction
    function automatic integer bval(input integer k, input integer n);
        bval = ((k * 7 + n * 3 + 2) & 15) - 8;
    endfunction
    function automatic integer golden(input integer r, input integer n);
        integer sum;
        integer index;
        begin
            sum = 0;
            for (index = 0; index < 10; index = index + 1)
                sum = sum +
                      (((r * 3 + index * 5 + 4) & 15) - 8) *
                      (((index * 7 + n * 3 + 2) & 15) - 8);
            golden = sum;
        end
    endfunction

    dual_int4_tiled_compute dut (
        .clk, .rst_n, .start, .M, .N, .K, .busy, .done, .error,
        .load_a_request, .load_b_request, .external_load_done,
        .external_load_preloaded(1'b0),
        .external_load_valid, .external_load_ready,
        .external_load_array, .external_load_matrix,
        .external_load_row, .external_load_mask, .external_load_data,
        .external_group_valid(1'b0), .external_group_ready(),
        .external_group_row(3'd0),
        .external_group_a_mask(8'd0), .external_group_a_data(32'd0),
        .external_group_b0_mask(8'd0), .external_group_b0_data(32'd0),
        .external_group_b1_mask(8'd0), .external_group_b1_data(32'd0),
        .seed_valid(1'b0), .seed_ready(), .seed_array(1'b0),
        .seed_addr(6'd0), .seed_data(32'sd0),
        .preserve_accumulator_seed(1'b0),
        .m0, .n0, .k0, .m_len, .k_len, .n_group_cols, .active_arrays,
        .output_valid, .output_ready, .output_array, .output_addr, .output_data
    );

    task automatic load_a_tile;
        logic [31:0] row_data;
        logic [7:0] row_mask;
        begin
            for (row = 0; row < m_len; row = row + 1) begin
                row_data = '0;
                row_mask = '0;
                for (col = 0; col < k_len; col = col + 1) begin
                    row_data[col*4 +: 4] = aval(m0+row, k0+col);
                    row_mask[col] = 1'b1;
                end
                external_load_valid <= 1;
                external_load_array <= 0;
                external_load_matrix <= 0;
                external_load_row <= row[2:0];
                external_load_mask <= row_mask;
                external_load_data <= row_data;
                do @(posedge clk); while (!external_load_ready);
            end
            external_load_valid <= 0;
        end
    endtask

    task automatic load_b_tiles;
        logic [31:0] row_data;
        logic [7:0] row_mask;
        begin
            for (array_id = 0; array_id < active_arrays; array_id = array_id + 1) begin
                for (row = 0; row < k_len; row = row + 1) begin
                    row_data = '0;
                    row_mask = '0;
                    for (col = 0; col < 8; col = col + 1) begin
                        if (array_id*8 + col < n_group_cols) begin
                            row_data[col*4 +: 4] =
                                bval(k0+row, n0+array_id*8+col);
                            row_mask[col] = 1'b1;
                        end
                    end
                    external_load_valid <= 1;
                    external_load_array <= array_id;
                    external_load_matrix <= 1;
                    external_load_row <= row[2:0];
                    external_load_mask <= row_mask;
                    external_load_data <= row_data;
                    do @(posedge clk); while (!external_load_ready);
                end
            end
            external_load_valid <= 0;
            @(posedge clk);
            external_load_done <= 1;
            @(posedge clk);
            external_load_done <= 0;
        end
    endtask

    always @(posedge clk) begin
        if (load_a_request) begin
            load_a_tile();
            load_b_tiles();
        end
        if (output_valid && output_ready) begin
            global_row = m0 + output_addr[5:3];
            global_col = n0 + output_array*8 + output_addr[2:0];
            expected = golden(global_row, global_col);
            if (output_data !== expected)
                $fatal(1, "C[%0d,%0d]=%0d expected %0d",
                       global_row, global_col, output_data, expected);
            outputs = outputs + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait (done);
        if (error) $fatal(1, "unexpected engine error");
        if (outputs != 13*19) $fatal(1, "output count %0d != 247", outputs);
        $display("Dual INT4 tiled engine golden trace passed: 13x19x10 with all tails");
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "timeout");
    end
endmodule
