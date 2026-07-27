module tb_gemm_dual_8x8_int4_core;
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
    logic load_valid;
    logic load_array;
    logic load_matrix;
    logic [5:0] load_addr;
    logic signed [3:0] load_data;
    logic load_vector_valid;
    logic load_vector_ready;
    logic load_vector_array;
    logic load_vector_matrix;
    logic [2:0] load_vector_row;
    logic [7:0] load_vector_mask;
    logic [31:0] load_vector_data;
    logic start;
    logic [1:0] share_mode;
    logic busy;
    logic done;
    logic result_array;
    logic [5:0] result_addr;
    logic signed [31:0] result_data;

    logic [3:0] a0_mem [0:63];
    logic [3:0] b0_mem [0:63];
    logic [3:0] a1_mem [0:63];
    logic [3:0] b1_mem [0:63];
    logic [31:0] c0_mem [0:63];
    logic [31:0] c1_ind_mem [0:63];
    logic [31:0] c1_sha_mem [0:63];
    logic [31:0] c1_shb_mem [0:63];
    integer errors;
    integer check_index;
    integer feed_cycle;

    always #5 clk = ~clk;

    gemm_dual_8x8_int4_core dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_valid(load_valid),
        .load_array(load_array),
        .load_matrix(load_matrix),
        .load_addr(load_addr),
        .load_data(load_data),
        .load_vector_valid(load_vector_valid),
        .load_vector_ready(load_vector_ready),
        .load_vector_array(load_vector_array),
        .load_vector_matrix(load_vector_matrix),
        .load_vector_row(load_vector_row),
        .load_vector_mask(load_vector_mask),
        .load_vector_data(load_vector_data),
        .load_group_valid(1'b0),
        .load_group_ready(),
        .load_group_row(3'd0),
        .load_group_a_mask(8'd0),
        .load_group_a_data(32'd0),
        .load_group_b0_mask(8'd0),
        .load_group_b0_data(32'd0),
        .load_group_b1_mask(8'd0),
        .load_group_b1_data(32'd0),
        .start(start),
        .share_mode(share_mode),
        .busy(busy),
        .done(done),
        .result_array(result_array),
        .result_addr(result_addr),
        .result_data(result_data),
        .result_row(3'd0),
        .result0_row(),
        .result1_row()
    );

    task load_tile(input logic selected_array, input logic selected_matrix);
        begin
            load_array = selected_array;
            load_matrix = selected_matrix;
            for (check_index = 0; check_index < 64;
                 check_index = check_index + 1) begin
                load_addr = check_index[5:0];
                case ({selected_array, selected_matrix})
                    2'b00: load_data = a0_mem[check_index];
                    2'b01: load_data = b0_mem[check_index];
                    2'b10: load_data = a1_mem[check_index];
                    default: load_data = b1_mem[check_index];
                endcase
                load_valid = 1'b1;
                @(posedge clk);
                #1;
            end
            load_valid = 1'b0;
        end
    endtask

    task check_mode(input logic [1:0] selected_mode);
        logic [31:0] expected;
        begin
            share_mode = selected_mode;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            wait (dut.state == 2'd2);
            for (feed_cycle = 0; feed_cycle < 26; feed_cycle = feed_cycle + 1) begin
                @(negedge clk);
                if (dut.state != 2'd2 || !dut.arrays.enable) begin
                    $error("INT4 dual core feed bubble mode=%0d cycle=%0d",
                           selected_mode, feed_cycle);
                    errors = errors + 1;
                end
                if (dut.compute_cycle !== feed_cycle[4:0]) begin
                    $error("INT4 dual core cadence mismatch mode=%0d expected=%0d actual=%0d",
                           selected_mode, feed_cycle, dut.compute_cycle);
                    errors = errors + 1;
                end
            end
            wait (done);
            @(posedge clk);

            for (check_index = 0; check_index < 64; check_index = check_index + 1) begin
                result_addr = check_index[5:0];
                result_array = 1'b0;
                #1;
                if (result_data !== c0_mem[check_index]) begin
                    $error("INT4 dual core array0 mismatch mode=%0d index=%0d", selected_mode, check_index);
                    errors = errors + 1;
                end

                result_array = 1'b1;
                case (selected_mode)
                    2'b01: expected = c1_sha_mem[check_index];
                    2'b10: expected = c1_shb_mem[check_index];
                    default: expected = c1_ind_mem[check_index];
                endcase
                #1;
                if (result_data !== expected) begin
                    $error("INT4 dual core array1 mismatch mode=%0d index=%0d", selected_mode, check_index);
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

        load_valid = 1'b0;
        load_array = 1'b0;
        load_matrix = 1'b0;
        load_addr = '0;
        load_data = '0;
        load_vector_valid = 1'b0;
        load_vector_array = 1'b0;
        load_vector_matrix = 1'b0;
        load_vector_row = '0;
        load_vector_mask = '0;
        load_vector_data = '0;
        start = 1'b0;
        share_mode = 2'b00;
        result_array = 1'b0;
        result_addr = '0;
        errors = 0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        load_tile(1'b0, 1'b0);
        load_tile(1'b0, 1'b1);
        load_tile(1'b1, 1'b0);
        load_tile(1'b1, 1'b1);
        check_mode(2'b00);
        check_mode(2'b01);
        check_mode(2'b10);

        if (errors != 0)
            $fatal(1, "INT4 dual GEMM core golden trace failed with %0d errors", errors);

        $display("INT4 dual GEMM core golden trace passed: independent/shared-A/shared-B, one wave set/cycle");
        $finish;
    end
endmodule
