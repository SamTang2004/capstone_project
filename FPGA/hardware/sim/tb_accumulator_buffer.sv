module tb_accumulator_buffer;
    localparam string OPS_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/accumulator_ops.hex";
    localparam string FINAL_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/accumulator_final_tile.hex";
    localparam int OP_COUNT = 768;
    localparam int ENTRY_COUNT = 256;

    logic clk;
    logic rst_n;
    logic op_valid;
    logic op_ready;
    logic op_clear;
    logic [7:0] op_addr;
    logic signed [31:0] op_data;
    logic signed [31:0] op_result;
    logic read_req;
    logic [7:0] read_addr;
    logic read_valid;
    logic signed [31:0] read_data;

    logic [63:0] ops [0:OP_COUNT-1];
    logic [31:0] final_words [0:ENTRY_COUNT-1];
    logic signed [31:0] model [0:ENTRY_COUNT-1];

    accumulator_buffer dut (
        .clk(clk),
        .rst_n(rst_n),
        .op_valid(op_valid),
        .op_ready(op_ready),
        .op_clear(op_clear),
        .op_addr(op_addr),
        .op_data(op_data),
        .op_result(op_result),
        .read_req(read_req),
        .read_addr(read_addr),
        .read_valid(read_valid),
        .read_data(read_data)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        logic signed [31:0] partial;
        logic signed [31:0] expected;

        $readmemh(OPS_PATH, ops);
        $readmemh(FINAL_PATH, final_words);

        rst_n = 1'b0;
        op_valid = 1'b0;
        op_clear = 1'b0;
        op_addr = '0;
        op_data = '0;
        read_req = 1'b0;
        read_addr = '0;
        for (int i = 0; i < ENTRY_COUNT; i++) begin
            model[i] = '0;
        end

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < OP_COUNT; i++) begin
            @(negedge clk);
            op_clear = ops[i][56];
            op_addr = ops[i][55:48];
            partial = ops[i][47:16];
            op_data = partial;
            op_valid = 1'b1;
            expected = op_clear ? partial : model[op_addr] + partial;
            @(posedge clk);
            if (!op_ready) $fatal(1, "accumulator did not accept op %0d", i);
            @(negedge clk);
            op_valid = 1'b0;
            @(posedge clk);
            #1;
            if (op_result !== expected) $fatal(1, "accumulator op mismatch at %0d addr=%0d expected=%0d got=%0d", i, op_addr, expected, op_result);
            model[op_addr] = expected;
        end

        @(negedge clk);
        op_valid = 1'b0;

        for (int i = 0; i < ENTRY_COUNT; i++) begin
            @(negedge clk);
            read_addr = i[7:0];
            read_req = 1'b1;
            @(posedge clk);
            #1;
            read_req = 1'b0;
            if (!read_valid) $fatal(1, "accumulator read response missing at %0d", i);
            #1;
            if (read_data !== signed'(final_words[i]) || model[i] !== signed'(final_words[i])) begin
                $fatal(1, "accumulator final mismatch addr=%0d expected=%0d read=%0d model=%0d",
                    i, signed'(final_words[i]), read_data, model[i]);
            end
        end

        $display("Accumulator buffer golden trace passed: %0d ops, %0d reads", OP_COUNT, ENTRY_COUNT);
        $finish;
    end

endmodule
