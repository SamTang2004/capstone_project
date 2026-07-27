`timescale 1ns/1ps
module tb_int4_output_postprocess;
    logic clk = 0;
    logic rst_n = 0;
    logic input_valid = 0;
    logic input_ready;
    logic signed [31:0] input_data = 0;
    logic input_array = 0;
    logic [5:0] input_addr = 0;
    logic output_valid;
    logic output_ready = 0;
    logic signed [3:0] output_data;
    logic output_array;
    logic [5:0] output_addr;

    always #5 clk = ~clk;

    int4_output_postprocess dut (
        .clk, .rst_n, .input_valid, .input_ready, .input_data,
        .input_array, .input_addr,
        .multiplier(32'sd1), .right_shift(6'd1),
        .output_zero_point(32'sd0),
        .activation_enable(1'b1), .leaky_mode(1'b1), .leaky_shift(3'd1),
        .output_valid, .output_ready, .output_data, .output_array, .output_addr
    );

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        if (!input_ready) $fatal(1, "postprocess not ready");
        input_valid <= 1;
        input_data <= -32'sd12;
        input_array <= 1;
        input_addr <= 6'd37;
        @(posedge clk);
        input_valid <= 0;
        wait (output_valid);
        if (output_data !== -4'sd3) $fatal(1, "got %0d expected -3", output_data);
        if (!output_array || output_addr != 37) $fatal(1, "metadata mismatch");
        repeat (4) begin
            @(posedge clk);
            if (!output_valid || output_data !== -4'sd3)
                $fatal(1, "output changed under backpressure");
        end
        output_ready <= 1;
        @(posedge clk);
        output_ready <= 0;
        @(posedge clk);
        if (output_valid) $fatal(1, "output did not retire");
        $display("INT4 output postprocess golden trace passed: truncation, LeakyReLU, backpressure");
        $finish;
    end
endmodule
