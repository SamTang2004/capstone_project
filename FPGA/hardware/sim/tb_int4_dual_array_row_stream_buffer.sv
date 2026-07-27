`timescale 1ns/1ps
module tb_int4_dual_array_row_stream_buffer;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic start_ready;
    logic input_valid = 0;
    logic input_ready;
    logic input_array = 0;
    logic input_matrix = 0;
    logic [2:0] input_row = 0;
    logic [7:0] input_mask = 0;
    logic [31:0] input_data = 0;
    logic output_valid;
    logic output_ready;
    logic [2:0] output_row;
    logic [7:0] output_a_mask, output_b0_mask, output_b1_mask;
    logic [31:0] output_a_data, output_b0_data, output_b1_data;
    logic done, error;
    logic [4:0] customer0_occupancy, customer1_occupancy;
    integer output_count = 0;
    integer cycle_count = 0;
    integer first_output_cycle = -1;
    integer last_input_cycle = -1;

    always #5 clk = ~clk;
    assign output_ready = cycle_count % 5 != 2;

    int4_dual_array_row_stream_buffer dut (
        .clk, .rst_n, .start, .start_ready,
        .active_arrays(2'd2),
        .expected_a_rows(8'hff), .expected_b_rows(8'hff),
        .input_valid, .input_ready, .input_array, .input_matrix,
        .input_row, .input_mask, .input_data,
        .output_valid, .output_ready, .output_row,
        .output_a_mask, .output_a_data,
        .output_b0_mask, .output_b0_data,
        .output_b1_mask, .output_b1_data,
        .done, .error, .customer0_occupancy, .customer1_occupancy
    );

    function automatic logic [31:0] packed_word(
        input integer stream,
        input integer row
    );
        logic [31:0] result;
        result = '0;
        for (int lane = 0; lane < 8; lane++)
            result[lane*4 +: 4] = (stream * 5 + row * 3 + lane) & 15;
        return result;
    endfunction

    task automatic send_vector(
        input logic array_id,
        input logic matrix_id,
        input integer row,
        input integer stream
    );
        begin
            @(negedge clk);
            input_array = array_id;
            input_matrix = matrix_id;
            input_row = row;
            input_mask = 8'hff;
            input_data = packed_word(stream, row);
            input_valid = 1;
            do @(posedge clk); while (!input_ready);
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (output_valid && output_ready) begin
            if (output_row != output_count ||
                output_a_mask != 8'hff ||
                output_b0_mask != 8'hff ||
                output_b1_mask != 8'hff ||
                output_a_data != packed_word(0, output_count) ||
                output_b0_data != packed_word(1, output_count) ||
                output_b1_data != packed_word(2, output_count))
                $fatal(1,
                    "stream buffer output %0d row=%0d A=%08x B0=%08x B1=%08x",
                    output_count, output_row, output_a_data,
                    output_b0_data, output_b1_data);
            if (first_output_cycle < 0)
                first_output_cycle <= cycle_count;
            output_count <= output_count + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        if (!start_ready)
            $fatal(1, "stream buffer not ready after reset");
        @(negedge clk);
        start = 1;
        @(posedge clk);
        @(negedge clk);
        start = 0;

        for (int row = 0; row < 8; row++) begin
            send_vector(0, 0, row, 0);
            send_vector(0, 1, row, 1);
            if (row != 0)
                send_vector(1, 1, row, 2);
        end
        // B1 row zero is intentionally last. No row may overtake row zero.
        if (output_count != 0)
            $fatal(1, "stream buffer reordered around missing B1 row zero");
        send_vector(1, 1, 0, 2);
        last_input_cycle = cycle_count;

        wait (done);
        @(posedge clk);
        if (error || output_count != 8)
            $fatal(1, "stream buffer final error=%0b outputs=%0d",
                   error, output_count);
        if (first_output_cycle < 0 || first_output_cycle > last_input_cycle + 2)
            $fatal(1, "stream buffer did not release the first joined row");
        $display(
            "INT4 dual-array row stream buffer golden trace passed: outputs=%0d first=%0d last-input=%0d",
            output_count, first_output_cycle, last_input_cycle);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "dual-array row stream buffer timeout");
    end
endmodule
