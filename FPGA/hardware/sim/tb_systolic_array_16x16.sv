module tb_systolic_array_16x16;
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_A.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_B.hex";
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_C.hex";

    logic clk = 0;
    logic rst_n = 0;
    logic clear;
    logic enable;
    logic signed [8*16-1:0] a_west_flat;
    logic signed [8*16-1:0] b_north_flat;
    logic signed [32*256-1:0] c_flat;

    logic [7:0] a_mem [0:255];
    logic [7:0] b_mem [0:255];
    logic [31:0] c_mem [0:255];

    integer cycle;
    integer i;
    integer r;
    integer c;
    integer k;
    integer errors;

    always #5 clk = ~clk;

    systolic_array_16x16 dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_west_flat(a_west_flat),
        .b_north_flat(b_north_flat),
        .c_flat(c_flat)
    );

    task automatic drive_cycle(input integer t);
        begin
            a_west_flat = '0;
            b_north_flat = '0;

            for (r = 0; r < 16; r = r + 1) begin
                k = t - r;
                if (k >= 0 && k < 16) begin
                    a_west_flat[r*8 +: 8] = a_mem[r*16 + k];
                end
            end

            for (c = 0; c < 16; c = c + 1) begin
                k = t - c;
                if (k >= 0 && k < 16) begin
                    b_north_flat[c*8 +: 8] = b_mem[k*16 + c];
                end
            end
        end
    endtask

    initial begin
        errors = 0;
        $readmemh(A_PATH, a_mem);
        $readmemh(B_PATH, b_mem);
        $readmemh(C_PATH, c_mem);

        clear = 1'b0;
        enable = 1'b0;
        a_west_flat = '0;
        b_north_flat = '0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        clear = 1'b1;
        @(posedge clk);
        #1;
        clear = 1'b0;

        // Drain the PE operand and multiplier registers after the final
        // skewed operands reach PE (15,15).
        for (cycle = 0; cycle <= 47; cycle = cycle + 1) begin
            @(negedge clk);
            drive_cycle(cycle);
            enable = 1'b1;
            @(posedge clk);
            #1;
        end

        enable = 1'b0;
        a_west_flat = '0;
        b_north_flat = '0;

        for (i = 0; i < 256; i = i + 1) begin
            if (c_flat[i*32 +: 32] !== c_mem[i]) begin
                $error("Systolic array mismatch index %0d: expected 0x%08h got 0x%08h", i, c_mem[i], c_flat[i*32 +: 32]);
                errors = errors + 1;
            end
        end

        if (errors != 0) begin
            $fatal(1, "Systolic array golden trace failed with %0d errors", errors);
        end

        $display("Systolic array golden trace passed: 256 outputs");
        $finish;
    end
endmodule
