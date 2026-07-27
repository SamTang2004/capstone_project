module tb_pe_int4_mac;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear;
    logic enable;
    logic signed [3:0] a_in;
    logic signed [3:0] b_in;
    logic signed [3:0] a_out;
    logic signed [3:0] b_out;
    logic signed [31:0] acc_out;

    integer a;
    integer b;
    integer errors;
    logic signed [31:0] expected;

    always #5 clk = ~clk;

    pe_int4_mac #(.USE_DSP(1'b0)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_in(a_in),
        .b_in(b_in),
        .a_out(a_out),
        .b_out(b_out),
        .acc_out(acc_out)
    );

    initial begin
        clear = 1'b0;
        enable = 1'b0;
        a_in = '0;
        b_in = '0;
        errors = 0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        for (a = -8; a <= 7; a = a + 1) begin
            for (b = -8; b <= 7; b = b + 1) begin
                @(negedge clk);
                clear = 1'b1;
                enable = 1'b0;
                @(posedge clk);
                #1;

                @(negedge clk);
                clear = 1'b0;
                enable = 1'b1;
                a_in = a[3:0];
                b_in = b[3:0];
                expected = a * b;
                @(posedge clk);
                #1;

                if (a_out !== a[3:0] || b_out !== b[3:0]) begin
                    $error("INT4 forwarding mismatch for a=%0d b=%0d", a, b);
                    errors = errors + 1;
                end

                @(negedge clk);
                enable = 1'b0;
                a_in = '0;
                b_in = '0;
                repeat (2) @(posedge clk);
                #1;

                if (acc_out !== expected) begin
                    $error(
                        "INT4 multiply mismatch a=%0d b=%0d expected=%0d got=%0d",
                        a, b, expected, acc_out
                    );
                    errors = errors + 1;
                end
            end
        end

        if (errors != 0)
            $fatal(1, "INT4 PE exhaustive test failed with %0d errors", errors);

        $display("INT4 PE golden trace passed: all 256 signed products");
        $finish;
    end
endmodule
