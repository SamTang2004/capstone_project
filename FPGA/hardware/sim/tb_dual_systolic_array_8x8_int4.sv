module tb_dual_systolic_array_8x8_int4;
    localparam string A0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_A0.hex";
    localparam string B0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_B0.hex";
    localparam string A1_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_A1.hex";
    localparam string B1_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_B1.hex";
    localparam string C0_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C0.hex";
    localparam string C1_IND_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C1_independent.hex";
    localparam string C1_SHA_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C1_shared_a.hex";
    localparam string C1_SHB_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_8x8_C1_shared_b.hex";

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear;
    logic enable;
    logic [1:0] share_mode;
    logic signed [4*8-1:0] a0_west_flat;
    logic signed [4*8-1:0] b0_north_flat;
    logic signed [4*8-1:0] a1_west_flat;
    logic signed [4*8-1:0] b1_north_flat;
    logic signed [32*64-1:0] c0_flat;
    logic signed [32*64-1:0] c1_flat;

    logic [3:0] a0_mem [0:63];
    logic [3:0] b0_mem [0:63];
    logic [3:0] a1_mem [0:63];
    logic [3:0] b1_mem [0:63];
    logic [31:0] c0_mem [0:63];
    logic [31:0] c1_ind_mem [0:63];
    logic [31:0] c1_sha_mem [0:63];
    logic [31:0] c1_shb_mem [0:63];

    integer row;
    integer col;
    integer k;
    integer errors;

    always #5 clk = ~clk;

    dual_systolic_array_8x8_int4 #(.USE_DSP(1'b0)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .share_mode(share_mode),
        .a0_west_flat(a0_west_flat),
        .b0_north_flat(b0_north_flat),
        .a1_west_flat(a1_west_flat),
        .b1_north_flat(b1_north_flat),
        .c0_flat(c0_flat),
        .c1_flat(c1_flat)
    );

    task automatic drive_cycle(input integer t);
        begin
            a0_west_flat = '0;
            b0_north_flat = '0;
            a1_west_flat = '0;
            b1_north_flat = '0;

            for (row = 0; row < 8; row = row + 1) begin
                k = t - row;
                if (k >= 0 && k < 8) begin
                    a0_west_flat[row*4 +: 4] = a0_mem[row*8 + k];
                    a1_west_flat[row*4 +: 4] = a1_mem[row*8 + k];
                end
            end

            for (col = 0; col < 8; col = col + 1) begin
                k = t - col;
                if (k >= 0 && k < 8) begin
                    b0_north_flat[col*4 +: 4] = b0_mem[k*8 + col];
                    b1_north_flat[col*4 +: 4] = b1_mem[k*8 + col];
                end
            end
        end
    endtask

    task automatic run_mode(input logic [1:0] selected_mode);
        integer cycle;
        integer i;
        logic [31:0] expected_c1;
        begin
            @(negedge clk);
            share_mode = selected_mode;
            clear = 1'b1;
            enable = 1'b0;
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
            a0_west_flat = '0;
            b0_north_flat = '0;
            a1_west_flat = '0;
            b1_north_flat = '0;

            for (i = 0; i < 64; i = i + 1) begin
                case (selected_mode)
                    2'b01: expected_c1 = c1_sha_mem[i];
                    2'b10: expected_c1 = c1_shb_mem[i];
                    default: expected_c1 = c1_ind_mem[i];
                endcase

                if (c0_flat[i*32 +: 32] !== c0_mem[i]) begin
                    $error("Dual INT4 array 0 mismatch mode=%0d index=%0d", selected_mode, i);
                    errors = errors + 1;
                end
                if (c1_flat[i*32 +: 32] !== expected_c1) begin
                    $error("Dual INT4 array 1 mismatch mode=%0d index=%0d", selected_mode, i);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        $readmemh(A0_PATH, a0_mem);
        $readmemh(B0_PATH, b0_mem);
        $readmemh(A1_PATH, a1_mem);
        $readmemh(B1_PATH, b1_mem);
        $readmemh(C0_PATH, c0_mem);
        $readmemh(C1_IND_PATH, c1_ind_mem);
        $readmemh(C1_SHA_PATH, c1_sha_mem);
        $readmemh(C1_SHB_PATH, c1_shb_mem);

        clear = 1'b0;
        enable = 1'b0;
        share_mode = 2'b00;
        a0_west_flat = '0;
        b0_north_flat = '0;
        a1_west_flat = '0;
        b1_north_flat = '0;
        errors = 0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        run_mode(2'b00);
        run_mode(2'b01);
        run_mode(2'b10);

        if (errors != 0)
            $fatal(1, "Dual INT4 systolic test failed with %0d errors", errors);

        $display("Dual INT4 systolic golden trace passed: independent/shared-A/shared-B");
        $finish;
    end
endmodule
