module tb_gemm_8x8_tile_core;
    localparam string TRACE_DIR = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces";
    logic clk = 0;
    logic rst_n = 0;
    logic start, weight_int4, input_valid, input_ready;
    logic [15:0] k_steps;
    logic [3:0] valid_rows, valid_cols;
    logic [63:0] activation_vector, packed_weight_vector;
    logic busy, result_valid, result_ready, error;
    logic [2047:0] accumulators;
    logic [127:0] w8_inputs [0:10];
    logic [127:0] w4_inputs [0:12];
    logic [31:0] w8_expected [0:63];
    logic [31:0] w4_expected [0:63];
    integer errors = 0;
    integer phase = 0;

    always #5 clk = ~clk;

    gemm_8x8_tile_core dut (.*);

    initial begin
        #100000;
        $fatal(1, "8x8 GEMM timeout phase=%0d state=%0d accepted=%0d ready=%0d result=%0d",
               phase, dut.state, dut.accepted_steps, input_ready, result_valid);
    end

    task automatic run_case(
        input logic use_w4,
        input integer reduction,
        input integer rows,
        input integer cols
    );
        logic [127:0] vector;
        logic signed [31:0] expected;
        logic signed [31:0] actual;
        begin
            @(negedge clk);
            weight_int4 = use_w4;
            k_steps = reduction;
            valid_rows = rows;
            valid_cols = cols;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (input_ready);
            phase = 1;
            for (int step = 0; step < reduction; step++) begin
                vector = use_w4 ? w4_inputs[step] : w8_inputs[step];
                @(negedge clk);
                activation_vector = vector[127:64];
                packed_weight_vector = vector[63:0];
                input_valid = 1'b1;
            end
            @(negedge clk);
            input_valid = 1'b0;
            activation_vector = '0;
            packed_weight_vector = '0;
            phase = 2;
            wait (result_valid);
            phase = 3;
            #1;
            for (int element = 0; element < 64; element++) begin
                expected = use_w4 ? w4_expected[element] : w8_expected[element];
                actual = accumulators[element*32 +: 32];
                if (actual !== expected) begin
                    $error("%s mismatch element=%0d expected=%0d actual=%0d",
                           use_w4 ? "W4A8" : "W8A8", element, expected, actual);
                    errors++;
                end
            end
            phase = 4;
            result_ready = 1'b1;
            @(posedge clk);
            #1;
            result_ready = 1'b0;
            phase = 5;
        end
    endtask

    initial begin
        $readmemh({TRACE_DIR, "/gemm8_w8_inputs.hex"}, w8_inputs);
        $readmemh({TRACE_DIR, "/gemm8_w8_expected.hex"}, w8_expected);
        $readmemh({TRACE_DIR, "/gemm8_w4_inputs.hex"}, w4_inputs);
        $readmemh({TRACE_DIR, "/gemm8_w4_expected.hex"}, w4_expected);
        start = 0;
        weight_int4 = 0;
        k_steps = 0;
        valid_rows = 0;
        valid_cols = 0;
        input_valid = 0;
        activation_vector = 0;
        packed_weight_vector = 0;
        result_ready = 0;
        // The UNISIM glbl model holds DSP48E1 in global reset for 100 ns.
        repeat (12) @(posedge clk);
        rst_n = 1;

        run_case(1'b0, 11, 8, 8);
        run_case(1'b1, 13, 5, 6);

        if (errors != 0) $fatal(1, "8x8 GEMM failed with %0d errors", errors);
        $display("8x8 W8A8 and packed-W4A8 golden trace passed");
        $finish;
    end
endmodule
