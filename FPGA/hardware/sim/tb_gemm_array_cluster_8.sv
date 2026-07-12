module tb_gemm_array_cluster_8;
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_A.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_B.hex";
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/gemm_16x16_C.hex";
    logic clk = 0, rst_n = 0, load_valid, load_matrix, start, busy, done;
    logic [2:0] load_array, result_array;
    logic [3:0] active_arrays;
    logic [7:0] load_addr, load_data, result_addr;
    logic signed [31:0] result_data;
    logic [7:0] a_mem [0:255], b_mem [0:255];
    logic [31:0] c_mem [0:255];
    integer errors;

    always #5 clk = ~clk;
    gemm_array_cluster_8 dut (
        .clk(clk), .rst_n(rst_n), .load_valid(load_valid), .load_matrix(load_matrix), .load_array(load_array),
        .load_addr(load_addr), .load_data(load_data), .start(start), .active_arrays(active_arrays), .busy(busy), .done(done),
        .result_array(result_array), .result_addr(result_addr), .result_data(result_data)
    );

    initial begin
        errors = 0; load_valid = 0; load_matrix = 0; load_array = 0; load_addr = 0; load_data = 0; start = 0; active_arrays = 8;
        $readmemh(A_PATH, a_mem); $readmemh(B_PATH, b_mem); $readmemh(C_PATH, c_mem);
        repeat (2) @(posedge clk); rst_n = 1;
        for (int i = 0; i < 256; i++) begin
            @(negedge clk); load_valid = 1; load_matrix = 0; load_addr = i; load_data = a_mem[i];
        end
        @(negedge clk); load_valid = 0;
        for (int array_idx = 0; array_idx < 8; array_idx++) begin
            for (int i = 0; i < 256; i++) begin
                @(negedge clk); load_valid = 1; load_matrix = 1; load_array = array_idx; load_addr = i; load_data = b_mem[i];
            end
        end
        @(negedge clk); load_valid = 0; start = 1;
        @(negedge clk); start = 0;
        wait (done);
        for (int array_idx = 0; array_idx < 8; array_idx++) begin
            result_array = array_idx;
            for (int i = 0; i < 256; i++) begin
                result_addr = i; #1;
                if (result_data !== c_mem[i]) begin
                    $error("Cluster result mismatch array=%0d index=%0d", array_idx, i);
                    errors = errors + 1;
                end
            end
        end
        if (errors != 0) $fatal(1, "Eight-array systolic cluster failed with %0d errors", errors);
        $display("Eight-array systolic cluster golden trace passed: 2048 outputs");
        $finish;
    end
endmodule
