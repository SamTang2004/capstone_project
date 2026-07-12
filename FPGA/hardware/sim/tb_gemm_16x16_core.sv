module tb_gemm_16x16_core;
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_A.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_B.hex";
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_C.hex";

    logic clk = 0;
    logic rst_n = 0;
    logic load_valid;
    logic load_matrix;
    logic [7:0] load_addr;
    logic [7:0] load_data;
    logic start;
    logic busy;
    logic done;
    logic [7:0] result_addr;
    logic [31:0] result_data;

    logic [7:0] a_mem [0:255];
    logic [7:0] b_mem [0:255];
    logic [31:0] c_mem [0:255];

    integer i;
    integer errors;

    always #5 clk = ~clk;

    gemm_16x16_core dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_valid(load_valid),
        .load_matrix(load_matrix),
        .load_addr(load_addr),
        .load_data(load_data),
        .start(start),
        .busy(busy),
        .done(done),
        .result_addr(result_addr),
        .result_data(result_data)
    );

    task automatic load_matrix_bytes(input logic matrix_select);
        begin
            load_matrix = matrix_select;
            for (i = 0; i < 256; i = i + 1) begin
                load_addr = i[7:0];
                load_data = matrix_select ? b_mem[i] : a_mem[i];
                load_valid = 1'b1;
                @(posedge clk);
                #1;
            end
            load_valid = 1'b0;
        end
    endtask

    initial begin
        errors = 0;
        $readmemh(A_PATH, a_mem);
        $readmemh(B_PATH, b_mem);
        $readmemh(C_PATH, c_mem);

        load_valid = 0;
        load_matrix = 0;
        load_addr = 0;
        load_data = 0;
        start = 0;
        result_addr = 0;

        repeat (2) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        load_matrix_bytes(1'b0);
        load_matrix_bytes(1'b1);

        start = 1'b1;
        @(posedge clk);
        #1;
        start = 1'b0;

        wait (done == 1'b1);
        @(posedge clk);

        for (i = 0; i < 256; i = i + 1) begin
            result_addr = i[7:0];
            #1;
            if (result_data !== c_mem[i]) begin
                $error("GEMM mismatch index %0d: expected 0x%08h got 0x%08h", i, c_mem[i], result_data);
                errors = errors + 1;
            end
        end

        if (errors != 0) begin
            $fatal(1, "GEMM core golden trace failed with %0d errors", errors);
        end

        $display("GEMM core golden trace passed: 256 outputs");
        $finish;
    end
endmodule
