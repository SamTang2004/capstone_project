module tb_activation_unfold_parallel;
    logic enable, leaky_mode;
    logic [2:0] leaky_shift;
    logic signed [16*8-1:0] input_flat, output_flat;
    integer errors;

    activation_unfold_parallel dut (
        .enable(enable), .leaky_mode(leaky_mode), .leaky_shift(leaky_shift), .input_flat(input_flat), .output_flat(output_flat)
    );

    initial begin
        errors = 0; enable = 1; leaky_shift = 2;
        for (int lane = 0; lane < 16; lane++) input_flat[lane*8 +: 8] = lane < 8 ? -8 + lane : lane - 8;
        leaky_mode = 0; #1;
        for (int lane = 0; lane < 16; lane++) begin
            if ($signed(output_flat[lane*8 +: 8]) != (lane < 8 ? 0 : lane - 8)) errors = errors + 1;
        end
        leaky_mode = 1; #1;
        for (int lane = 0; lane < 16; lane++) begin
            if ($signed(output_flat[lane*8 +: 8]) != (lane < 8 ? ((-8 + lane) >>> 2) : lane - 8)) errors = errors + 1;
        end
        if (errors != 0) $fatal(1, "Parallel activation failed with %0d errors", errors);
        $display("Parallel ReLU and Leaky ReLU golden trace passed: 16 lanes");
        $finish;
    end
endmodule
