`timescale 1ns/1ps
module tb_int4_streaming_scalar_row_packer;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic finish = 0;
    logic scalar_valid = 0;
    logic [5:0] scalar_addr = 0;
    logic signed [3:0] scalar_data = 0;
    logic vector_valid;
    logic vector_ready = 1;
    logic vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;
    logic done, error;
    integer output_count = 0;
    integer first_output_cycle = -1;
    integer finish_cycle = -1;
    integer cycle_count = 0;

    always #5 clk = ~clk;

    int4_streaming_scalar_row_packer dut (
        .clk, .rst_n, .start, .finish,
        .transaction_array(1'b0), .transaction_matrix(1'b0),
        .expected_rows(4'd3), .expected_lane_mask(8'h1f),
        .scalar_valid, .scalar_addr, .scalar_data,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data, .done, .error
    );

    function automatic logic [31:0] expected_word(input integer row);
        logic [31:0] result;
        result = '0;
        for (int lane = 0; lane < 5; lane++)
            result[lane*4 +: 4] = (row * 5 + lane) & 15;
        return result;
    endfunction

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (vector_valid && vector_ready) begin
            if (vector_row != output_count ||
                vector_mask != 8'h1f ||
                vector_data != expected_word(output_count))
                $fatal(1,
                    "streaming packer row %0d got row=%0d mask=%02x data=%08x",
                    output_count, vector_row, vector_mask, vector_data);
            if (first_output_cycle < 0)
                first_output_cycle <= cycle_count;
            output_count <= output_count + 1;
        end
    end

    task automatic send_scalar(input integer row, input integer lane);
        begin
            @(negedge clk);
            scalar_addr = row * 8 + lane;
            scalar_data = (row * 5 + lane) & 15;
            scalar_valid = 1;
            @(posedge clk);
            @(negedge clk);
            scalar_valid = 0;
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(posedge clk);
        @(negedge clk);
        start = 0;

        for (int row = 0; row < 3; row++)
            for (int lane = 0; lane < 5; lane++)
                send_scalar(row, lane);

        finish_cycle = cycle_count;
        finish = 1;
        @(posedge clk);
        @(negedge clk);
        finish = 0;
        wait (done);
        @(posedge clk);

        if (error || output_count != 3)
            $fatal(1, "packer final error=%0b outputs=%0d", error, output_count);
        if (first_output_cycle < 0 || first_output_cycle >= finish_cycle)
            $fatal(1,
                "first row was not streamed before finish: first=%0d finish=%0d",
                first_output_cycle, finish_cycle);
        $display(
            "INT4 streaming scalar row packer golden trace passed: rows=%0d first-output=%0d finish=%0d",
            output_count, first_output_cycle, finish_cycle);
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "streaming packer timeout");
    end
endmodule
