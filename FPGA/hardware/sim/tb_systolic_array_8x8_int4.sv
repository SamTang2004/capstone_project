module tb_systolic_array_8x8_int4;
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_A0.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_B0.hex";
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C0.hex";

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear;
    logic enable;
    logic signed [4*8-1:0] a_west_flat;
    logic signed [4*8-1:0] b_north_flat;
    logic signed [32*64-1:0] c_flat;

    logic [3:0] a_mem [0:63];
    logic [3:0] b_mem [0:63];
    logic [31:0] c_mem [0:63];

    integer cycle;
    integer i;
    integer row;
    integer col;
    integer k;
    integer errors;

    always #5 clk = ~clk;

    systolic_array_8x8_int4 #(.USE_DSP(1'b0)) dut (
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

            for (row = 0; row < 8; row = row + 1) begin
                k = t - row;
                if (k >= 0 && k < 8)
                    a_west_flat[row*4 +: 4] = a_mem[row*8 + k];
            end

            for (col = 0; col < 8; col = col + 1) begin
                k = t - col;
                if (k >= 0 && k < 8)
                    b_north_flat[col*4 +: 4] = b_mem[k*8 + col];
            end
        end
    endtask

    initial begin
        $readmemh(A_PATH, a_mem);
        $readmemh(B_PATH, b_mem);
        $readmemh(C_PATH, c_mem);

        clear = 1'b0;
        enable = 1'b0;
        a_west_flat = '0;
        b_north_flat = '0;
        errors = 0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        clear = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        clear = 1'b0;

        for (cycle = 0; cycle <= 23; cycle = cycle + 1) begin
            drive_cycle(cycle);
            enable = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
        end

        enable = 1'b0;
        a_west_flat = '0;
        b_north_flat = '0;

        for (i = 0; i < 64; i = i + 1) begin
            if (c_flat[i*32 +: 32] !== c_mem[i]) begin
                $error(
                    "INT4 8x8 mismatch index %0d: expected 0x%08h got 0x%08h",
                    i, c_mem[i], c_flat[i*32 +: 32]
                );
                errors = errors + 1;
            end
        end

        if (errors != 0)
            $fatal(1, "INT4 8x8 systolic test failed with %0d errors", errors);

        $display("INT4 8x8 systolic golden trace passed: 64 outputs");
        $finish;
    end
endmodule
