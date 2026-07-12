module tb_memory_io_driver;
    localparam string INITIAL_DDR_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/memory_io_initial_ddr.hex";
    localparam string READS_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/memory_io_reads.hex";
    localparam string WRITES_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/memory_io_writes.hex";
    localparam string FINAL_DDR_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/memory_io_final_ddr.hex";

    localparam int MEM_BYTES = 256;
    localparam int READ_COUNT = 6;
    localparam int WRITE_COUNT = 4;

    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic read_req_valid;
    logic read_req_ready;
    logic [31:0] read_req_addr;
    logic read_data_valid;
    logic read_data_ready;
    logic [7:0] read_data;

    logic write_req_valid;
    logic write_req_ready;
    logic [31:0] write_req_addr;
    logic [31:0] write_req_data;
    logic [3:0] write_req_strb;

    logic mem_cmd_valid;
    logic mem_cmd_ready;
    logic mem_cmd_write;
    logic [31:0] mem_cmd_addr;
    logic [31:0] mem_cmd_wdata;
    logic [3:0] mem_cmd_wstrb;
    logic mem_read_valid;
    logic mem_read_ready;
    logic [31:0] mem_read_data;

    logic busy;
    logic error;

    logic [7:0] ddr [0:MEM_BYTES-1];
    logic [7:0] expected_final_ddr [0:MEM_BYTES-1];
    logic [63:0] read_trace [0:READ_COUNT-1];
    logic [95:0] write_trace [0:WRITE_COUNT-1];

    int cycle;
    int read_issue_idx;
    int write_issue_idx;
    int read_resp_idx;
    int errors;
    int response_stall_countdown;
    logic pending_mem_read;
    logic [31:0] pending_mem_addr;

    always #5 clk = ~clk;

    memory_io_driver dut (
        .clk(clk),
        .rst_n(rst_n),
        .read_req_valid(read_req_valid),
        .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr),
        .read_data_valid(read_data_valid),
        .read_data_ready(read_data_ready),
        .read_data(read_data),
        .write_req_valid(write_req_valid),
        .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr),
        .write_req_data(write_req_data),
        .write_req_strb(write_req_strb),
        .mem_cmd_valid(mem_cmd_valid),
        .mem_cmd_ready(mem_cmd_ready),
        .mem_cmd_write(mem_cmd_write),
        .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_wdata(mem_cmd_wdata),
        .mem_cmd_wstrb(mem_cmd_wstrb),
        .mem_read_valid(mem_read_valid),
        .mem_read_ready(mem_read_ready),
        .mem_read_data(mem_read_data),
        .busy(busy),
        .error(error)
    );

    function automatic logic [31:0] load_word_le(input logic [31:0] addr);
        begin
            load_word_le = {ddr[addr + 3], ddr[addr + 2], ddr[addr + 1], ddr[addr]};
        end
    endfunction

    task automatic store_word_le(input logic [31:0] addr, input logic [31:0] data, input logic [3:0] wstrb);
        begin
            for (int lane = 0; lane < 4; lane++) begin
                if (wstrb[lane]) begin
                    ddr[addr + lane] = data[8*lane +: 8];
                end
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            mem_cmd_ready <= 1'b0;
            mem_read_valid <= 1'b0;
            mem_read_data <= '0;
            pending_mem_read <= 1'b0;
            pending_mem_addr <= '0;
        end else begin
            cycle <= cycle + 1;
            mem_cmd_ready <= 1'b1;
            mem_read_valid <= pending_mem_read;
            mem_read_data <= load_word_le(pending_mem_addr);
            pending_mem_read <= 1'b0;

            if (mem_cmd_valid && mem_cmd_ready) begin
                if (mem_cmd_write) begin
                    if (mem_cmd_addr[1:0] != 2'b00 || mem_cmd_wstrb != 4'hF) begin
                        $error("Memory command write alignment/strobe mismatch addr=0x%08h wstrb=0x%01h", mem_cmd_addr, mem_cmd_wstrb);
                        errors++;
                    end
                    store_word_le(mem_cmd_addr, mem_cmd_wdata, mem_cmd_wstrb);
                end else begin
                    if (mem_cmd_addr[1:0] != 2'b00) begin
                        $error("Memory command read was not aligned: 0x%08h", mem_cmd_addr);
                        errors++;
                    end
                    pending_mem_read <= 1'b1;
                    pending_mem_addr <= mem_cmd_addr;
                end
            end
        end
    end

    always_comb begin
        read_req_valid = 1'b0;
        read_req_addr = '0;
        if (read_issue_idx < READ_COUNT && cycle >= read_trace[read_issue_idx][63:48]) begin
            read_req_valid = 1'b1;
            read_req_addr = read_trace[read_issue_idx][47:16];
        end

        write_req_valid = 1'b0;
        write_req_addr = '0;
        write_req_data = '0;
        write_req_strb = 4'hf;
        if (write_issue_idx < WRITE_COUNT && cycle >= write_trace[write_issue_idx][95:80]) begin
            write_req_valid = 1'b1;
            write_req_addr = write_trace[write_issue_idx][79:48];
            write_req_data = write_trace[write_issue_idx][47:16];
        end
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (read_req_valid && read_req_ready) begin
                read_issue_idx++;
                response_stall_countdown <= read_trace[read_issue_idx][15:8];
            end

            if (write_req_valid && write_req_ready) begin
                write_issue_idx++;
            end

            if (response_stall_countdown > 0) begin
                read_data_ready <= 1'b0;
                response_stall_countdown--;
            end else begin
                read_data_ready <= 1'b1;
            end

            if (read_data_valid && read_data_ready) begin
                if (read_resp_idx >= READ_COUNT) begin
                    $error("Unexpected extra read response 0x%02h", read_data);
                    errors++;
                end else if (read_data !== read_trace[read_resp_idx][7:0]) begin
                    $error("Read response mismatch idx=%0d expected=0x%02h got=0x%02h",
                        read_resp_idx, read_trace[read_resp_idx][7:0], read_data);
                    errors++;
                end
                read_resp_idx++;
            end
        end
    end

    initial begin
        $readmemh(INITIAL_DDR_PATH, ddr);
        $readmemh(FINAL_DDR_PATH, expected_final_ddr);
        $readmemh(READS_PATH, read_trace);
        $readmemh(WRITES_PATH, write_trace);

        errors = 0;
        read_issue_idx = 0;
        write_issue_idx = 0;
        read_resp_idx = 0;
        response_stall_countdown = 0;
        read_data_ready = 1'b1;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        repeat (120) @(posedge clk);

        if (error) begin
            $error("memory_io_driver reported error");
            errors++;
        end
        if (read_issue_idx != READ_COUNT || read_resp_idx != READ_COUNT || write_issue_idx != WRITE_COUNT) begin
            $error("Memory IO count mismatch reads issued=%0d responses=%0d writes=%0d",
                read_issue_idx, read_resp_idx, write_issue_idx);
            errors++;
        end
        for (int i = 0; i < MEM_BYTES; i++) begin
            if (ddr[i] !== expected_final_ddr[i]) begin
                $error("Final DDR mismatch byte %0d expected=0x%02h got=0x%02h", i, expected_final_ddr[i], ddr[i]);
                errors++;
            end
        end

        if (errors != 0) begin
            $fatal(1, "Memory IO driver golden trace failed with %0d errors", errors);
        end

        $display("Memory IO driver golden trace passed: %0d reads, %0d writes", READ_COUNT, WRITE_COUNT);
        $finish;
    end

endmodule
