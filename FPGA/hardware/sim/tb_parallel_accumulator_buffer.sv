`timescale 1ns/1ps
module tb_parallel_accumulator_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic row_valid = 1'b0;
    logic row_ready;
    logic row_clear = 1'b0;
    logic [1:0] row_active_arrays = 2;
    logic [2:0] row_addr = '0;
    logic signed [255:0] row_data0 = '0;
    logic signed [255:0] row_data1 = '0;
    logic row_commit;
    logic [2:0] row_commit_addr;
    logic seed_valid = 1'b0;
    logic seed_ready;
    logic seed_array = 1'b0;
    logic [5:0] seed_addr = '0;
    logic signed [31:0] seed_data = '0;
    logic read_req = 1'b0;
    logic read_array = 1'b0;
    logic [5:0] read_addr = '0;
    logic read_valid;
    logic signed [31:0] read_data;
    integer cycle_count = 0;
    integer first_accept_cycle;
    integer final_commit_cycle;

    always #5 clk = ~clk;
    always @(posedge clk)
        if (rst_n) cycle_count <= cycle_count + 1;

    parallel_accumulator_buffer dut (
        .clk, .rst_n,
        .row_valid, .row_ready, .row_clear, .row_active_arrays,
        .row_addr, .row_data0, .row_data1,
        .row_commit, .row_commit_addr,
        .seed_valid, .seed_ready, .seed_array, .seed_addr, .seed_data,
        .read_req, .read_array, .read_addr, .read_valid, .read_data
    );

    task automatic send_pass(input logic clear_pass,
                             input integer increment);
        integer row;
        integer col;
        begin
            for (row = 0; row < 8; row++) begin
                @(negedge clk);
                row_valid = 1'b1;
                row_clear = clear_pass;
                row_addr = row[2:0];
                for (col = 0; col < 8; col++) begin
                    row_data0[col*32 +: 32] =
                        clear_pass ? row*8 + col : increment;
                    row_data1[col*32 +: 32] =
                        clear_pass ? -(row*8 + col) : increment;
                end
                @(posedge clk);
                if (!row_ready)
                    $fatal(1, "parallel accumulator stalled at row %0d", row);
                if (row == 0)
                    first_accept_cycle = cycle_count;
            end
            @(negedge clk);
            row_valid = 1'b0;
            wait (row_commit && row_commit_addr == 7);
            final_commit_cycle = cycle_count;
            @(posedge clk);
        end
    endtask

    task automatic check_value(input logic selected_array,
                               input integer address,
                               input integer expected);
        begin
            @(negedge clk);
            read_req = 1'b1;
            read_array = selected_array;
            read_addr = address[5:0];
            @(posedge clk);
            @(negedge clk);
            read_req = 1'b0;
            wait (read_valid);
            if (read_data !== expected)
                $fatal(1, "array=%0d addr=%0d got=%0d expected=%0d",
                       selected_array, address, read_data, expected);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        send_pass(1'b1, 0);
        if (final_commit_cycle - first_accept_cycle > 10)
            $fatal(1, "parallel clear pass took %0d cycles",
                   final_commit_cycle - first_accept_cycle);

        send_pass(1'b0, 3);
        if (final_commit_cycle - first_accept_cycle > 10)
            $fatal(1, "parallel accumulate pass took %0d cycles",
                   final_commit_cycle - first_accept_cycle);

        for (integer address = 0; address < 64; address++) begin
            check_value(1'b0, address, address + 3);
            check_value(1'b1, address, -address + 3);
        end

        @(negedge clk);
        seed_array = 1'b1;
        seed_addr = 6'd17;
        seed_data = 32'sd12345;
        seed_valid = 1'b1;
        @(posedge clk);
        if (!seed_ready)
            $fatal(1, "seed port unexpectedly stalled");
        @(negedge clk);
        seed_valid = 1'b0;
        check_value(1'b1, 17, 12345);

        $display("Parallel accumulator golden trace passed: two arrays, 16 values/cycle, <=10-cycle tile accumulation");
        $finish;
    end

    initial begin
        repeat (1000) @(posedge clk);
        $fatal(1, "parallel accumulator timeout");
    end
endmodule
