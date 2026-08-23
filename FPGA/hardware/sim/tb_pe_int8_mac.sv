module tb_pe_int8_mac;
    localparam string TRACE_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/pe_int8_mac_trace.hex";

    logic clk = 0;
    logic rst_n = 0;
    logic clear;
    logic enable;
    logic signed [7:0] a;
    logic signed [7:0] b;
    logic signed [7:0] a_out;
    logic signed [7:0] b_out;
    logic signed [31:0] acc;

    logic [55:0] trace_mem [0:255];
    integer trace_count;
    integer errors;
    integer i;

    logic expected_clear;
    logic expected_enable;
    logic signed [7:0] expected_a;
    logic signed [7:0] expected_b;
    logic signed [31:0] expected_acc;

    always #5 clk = ~clk;

    pe_int8_mac #(.USE_DSP(1'b1)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_in(a),
        .b_in(b),
        .a_out(a_out),
        .b_out(b_out),
        .acc_out(acc)
    );

    initial begin
        trace_count = 0;
        errors = 0;
        $readmemh(TRACE_PATH, trace_mem);

        for (i = 0; i < 256; i = i + 1) begin
            if (trace_mem[i] !== 'x) begin
                trace_count = trace_count + 1;
            end
        end

        if (trace_count == 0) begin
            $fatal(1, "No PE golden trace rows loaded from %s", TRACE_PATH);
        end

        rst_n = 0;
        clear = 0;
        enable = 0;
        a = 0;
        b = 0;
        // The UNISIM glbl model holds DSP48E1 in global reset for 100 ns.
        repeat (12) @(posedge clk);
        rst_n = 1;

        for (i = 0; i < trace_count; i = i + 1) begin
            expected_clear = trace_mem[i][52];
            expected_enable = trace_mem[i][48];
            expected_a = trace_mem[i][47:40];
            expected_b = trace_mem[i][39:32];
            expected_acc = trace_mem[i][31:0];
            clear = expected_clear;
            enable = expected_enable;
            a = expected_a;
            b = expected_b;
            @(posedge clk);
            #1;
            if (acc !== expected_acc) begin
                $error(
                    "PE mismatch row %0d: clear=%0d enable=%0d a=%0d b=%0d expected_acc=%0d got=%0d",
                    i,
                    expected_clear,
                    expected_enable,
                    expected_a,
                    expected_b,
                    expected_acc,
                    acc
                );
                errors = errors + 1;
            end
        end

        if (errors != 0) begin
            $fatal(1, "PE golden trace failed with %0d errors", errors);
        end

        $display("PE golden trace passed: %0d rows", trace_count);

        $finish;
    end
endmodule
