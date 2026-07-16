module tb_avgpool_ddr;
    logic clk = 0, rst_n = 0, start;
    logic [31:0] input_base_addr, output_base_addr;
    logic [15:0] n, channels, input_h, input_w, output_h, output_w;
    logic [7:0] kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    logic busy, done, error, read_req_valid, read_req_ready, read_data_valid, read_data_ready, write_req_valid, write_req_ready;
    logic [31:0] read_req_addr, write_req_addr, write_req_data;
    logic [7:0] read_data; logic [3:0] write_req_strb;
    logic [7:0] mem [0:1023]; logic pending; logic [31:0] pending_addr; integer cycles, start_cycle, first_request_cycle;
    always #5 clk = ~clk;
    assign read_req_ready = !pending;
    assign write_req_ready = 1'b1;
    avgpool_ddr dut (.*);
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (start) begin start_cycle <= cycles; first_request_cycle <= -1; end
        if (read_req_valid && read_req_ready && first_request_cycle < 0) first_request_cycle <= cycles;
        read_data_valid <= 0;
        if (pending) begin read_data <= mem[pending_addr]; read_data_valid <= 1; if (read_data_ready) pending <= 0; end
        if (read_req_valid && read_req_ready) begin pending <= 1; pending_addr <= read_req_addr; end
        if (write_req_valid && write_req_ready) for (int i = 0; i < 4; i++) if (write_req_strb[i]) mem[write_req_addr+i] <= write_req_data[i*8 +: 8];
    end
    task automatic run_op;
        begin @(negedge clk); start = 1; @(negedge clk); start = 0; while (!done) @(posedge clk); if (error) $fatal(1, "AveragePool error"); if (first_request_cycle - start_cycle < 23) $fatal(1, "AveragePool pipeline latency was %0d cycles, expected at least 23", first_request_cycle - start_cycle); @(posedge clk); end
    endtask
    task automatic check_value(input int addr, input integer value);
        if ($signed(mem[addr]) !== value) $fatal(1, "average mismatch addr=%0d got=%0d expected=%0d", addr, $signed(mem[addr]), value);
    endtask
    initial begin
        cycles=0; start_cycle=-1; first_request_cycle=-1; start=0; input_base_addr=32'h100; output_base_addr=32'h200; n=1; channels=1; input_h=0; input_w=0; output_h=0; output_w=0; kernel_h=0; kernel_w=0; stride_h=0; stride_w=0; pad_h=0; pad_w=0; pending=0; read_data_valid=0; read_data=0; pending_addr=0;
        for (int i=0;i<1024;i++) mem[i]=0;
        repeat(3) @(posedge clk); rst_n=1;
        // [-7,-2;8,4] -> 0 (sum=3), [-2,3;4,-5] -> 0,
        // [8,4;6,9] -> 6, and [4,-5;9,-1] -> 1.
        mem[256]=-7; mem[257]=-2; mem[258]=3; mem[259]=8; mem[260]=4; mem[261]=-5; mem[262]=6; mem[263]=9; mem[264]=-1;
        n=1; channels=1; input_h=3; input_w=3; output_h=2; output_w=2; kernel_h=2; kernel_w=2; stride_h=1; stride_w=1; pad_h=0; pad_w=0;
        run_op();
        check_value(512,0); check_value(513,0); check_value(514,6); check_value(515,1);
        // GlobalAveragePool uses the identical engine with one full-spatial window: sum=15, average=1.
        output_h=1; output_w=1; kernel_h=3; kernel_w=3; stride_h=1; stride_w=1;
        run_op(); check_value(512,1);
        // Full 16x16 windows expose accidental zero extension: channel 0 is
        // all -8 and channel 1 alternates -7/+3, yielding -8 and -2.
        input_base_addr=32'h000; output_base_addr=32'h300; channels=2; input_h=16; input_w=16; output_h=1; output_w=1; kernel_h=16; kernel_w=16; stride_h=16; stride_w=16;
        for (int i=0;i<256;i++) begin mem[i]=-8; mem[256+i]=(i[0] ? 3 : -7); end
        run_op(); check_value(768,-8); check_value(769,-2);
        $display("tb_avgpool_ddr golden trace passed"); $finish;
    end
endmodule
