`timescale 1ns/1ps
module tb_int4_cnn_tile_pipeline;
    localparam string A0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_A0.hex";
    localparam string B0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_B0.hex";
    localparam string A1_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_A1.hex";
    localparam string B1_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_B1.hex";
    localparam string C0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C0.hex";
    localparam string C1_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C1_independent.hex";

    logic clk = 0, rst_n = 0, load_valid, load_array, load_matrix, start;
    logic [5:0] load_addr;
    logic signed [3:0] load_data;
    logic bias_load_valid, bias_load_array;
    logic [2:0] bias_load_col;
    logic signed [31:0] bias_load_data;
    logic [1:0] share_mode;
    logic signed [31:0] multiplier, output_zero_point;
    logic [5:0] right_shift;
    logic leaky_mode;
    logic [2:0] leaky_shift;
    logic busy, done, output_valid, output_array;
    logic [5:0] output_addr;
    logic signed [3:0] output_data;
    logic [3:0] a0 [0:63], b0 [0:63], a1 [0:63], b1 [0:63];
    logic signed [31:0] c0 [0:63], c1 [0:63];
    integer i, outputs;

    always #5 clk = ~clk;
    int4_cnn_tile_pipeline dut (.*);

    function automatic logic signed [3:0] expected_value(
        input logic selected_array, input logic [5:0] addr
    );
        logic signed [63:0] scaled;
        begin
            scaled = ((selected_array ? c1[addr] : c0[addr]) +
                      (selected_array ? 32'sd2 : -32'sd1)) >>> 3;
            if (scaled < 0) expected_value = 0;
            else if (scaled > 7) expected_value = 7;
            else expected_value = scaled[3:0];
        end
    endfunction

    task automatic load_tile(input logic array_sel, input logic matrix_sel);
        begin
            load_array = array_sel;
            load_matrix = matrix_sel;
            for (i = 0; i < 64; i++) begin
                load_addr = i[5:0];
                case ({array_sel, matrix_sel})
                    2'b00: load_data = a0[i];
                    2'b01: load_data = b0[i];
                    2'b10: load_data = a1[i];
                    default: load_data = b1[i];
                endcase
                load_valid = 1'b1;
                @(posedge clk);
                #1;
            end
            load_valid = 1'b0;
        end
    endtask

    initial begin
        $readmemh(A0_PATH, a0); $readmemh(B0_PATH, b0);
        $readmemh(A1_PATH, a1); $readmemh(B1_PATH, b1);
        $readmemh(C0_PATH, c0); $readmemh(C1_PATH, c1);
        load_valid = 0; load_array = 0; load_matrix = 0; load_addr = 0; load_data = 0;
        bias_load_valid = 0; bias_load_array = 0; bias_load_col = 0; bias_load_data = 0;
        start = 0; share_mode = 0; multiplier = 1; right_shift = 3;
        output_zero_point = 0; leaky_mode = 0; leaky_shift = 2; outputs = 0;
        repeat (3) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);
        load_tile(0, 0); load_tile(0, 1); load_tile(1, 0); load_tile(1, 1);
        for (i = 0; i < 8; i++) begin
            @(negedge clk); bias_load_valid = 1; bias_load_array = 0; bias_load_col = i; bias_load_data = -1;
            @(negedge clk); bias_load_array = 1; bias_load_data = 2;
        end
        @(negedge clk); bias_load_valid = 0; start = 1;
        @(negedge clk); start = 0;
        while (!done) begin
            @(posedge clk); #1;
            if (output_valid) begin
                if (output_data !== expected_value(output_array, output_addr))
                    $fatal(1, "CNN tile mismatch array=%0d addr=%0d got=%0d expected=%0d",
                           output_array, output_addr, output_data,
                           expected_value(output_array, output_addr));
                outputs = outputs + 1;
            end
        end
        if (outputs != 128) $fatal(1, "Expected 128 outputs, got %0d", outputs);
        $display("tb_int4_cnn_tile_pipeline golden trace passed");
        $finish;
    end
endmodule
