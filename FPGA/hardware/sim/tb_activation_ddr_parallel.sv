module tb_activation_ddr_parallel;
    logic clk = 0, rst_n = 0, start, leaky; logic [2:0] shift; logic [31:0] in_base, out_base, count;
    logic busy, done, error, rv, rr, rdv, rdr, wv, wr; logic [31:0] ra, wa, wd; logic [7:0] rd; logic [3:0] ws;
    logic signed [7:0] mem [0:63]; integer errors, cycles, start_cycle, first_request_cycle;
    always #5 clk = ~clk;
    activation_ddr_parallel dut(.clk(clk),.rst_n(rst_n),.start(start),.leaky_mode(leaky),.leaky_shift(shift),.input_base_addr(in_base),.output_base_addr(out_base),.element_count(count),.busy(busy),.done(done),.error(error),.read_req_valid(rv),.read_req_ready(rr),.read_req_addr(ra),.read_data_valid(rdv),.read_data_ready(rdr),.read_data(rd),.write_req_valid(wv),.write_req_ready(wr),.write_req_addr(wa),.write_req_data(wd),.write_req_strb(ws));
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (start) begin start_cycle <= cycles; first_request_cycle <= -1; end
        if (rv && rr && first_request_cycle < 0) first_request_cycle <= cycles;
        rr <= 1; wr <= 1; rdv <= rv; rd <= mem[ra];
        if (wv && wr) for (int i=0;i<4;i++) if (ws[i]) mem[wa+i] <= wd[i*8 +: 8];
    end
    initial begin
        errors=0; cycles=0; start_cycle=-1; first_request_cycle=-1; start=0; leaky=1; shift=2; in_base=0; out_base=32; count=17; rr=0; wr=0; rdv=0; rd=0;
        for (int i=0;i<17;i++) mem[i] = i-9;
        repeat(3) @(posedge clk); rst_n=1; @(posedge clk); start=1; @(posedge clk); start=0;
        wait(done); #1;
        if(first_request_cycle-start_cycle<1)$fatal(1,"Activation pipeline latency was %0d cycles, expected at least 1",first_request_cycle-start_cycle);
        for (int i=0;i<17;i++) if (mem[32+i] !== (i-9 < 0 ? ((i-9) >>> 2) : i-9)) errors++;
        if(error || errors) $fatal(1,"activation DDR trace failed errors=%0d",errors);
        $display("Parallel DDR Leaky ReLU golden trace passed: 17 values"); $finish;
    end
endmodule
