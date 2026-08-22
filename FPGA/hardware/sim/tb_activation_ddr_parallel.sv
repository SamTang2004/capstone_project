module tb_activation_ddr_parallel;
    logic clk = 0, rst_n = 0, start, leaky;
    logic [2:0] shift;
    logic [31:0] in_base, out_base, count;
    logic busy, done, error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid, read_word_ready, read_word_last;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic write_req_valid, write_req_ready;
    logic [31:0] write_req_addr, write_req_data;
    logic [3:0] write_req_strb;
    logic signed [7:0] mem [0:63];
    logic response_pending;
    logic [31:0] response_addr;
    logic [4:0] response_remaining;
    integer errors, request_count, cycles, start_cycle, done_cycle;

    always #5 clk = ~clk;
    assign read_req_ready = !response_pending && !read_word_valid;
    assign write_req_ready = 1'b1;

    activation_ddr_parallel dut (
        .clk(clk), .rst_n(rst_n), .start(start), .leaky_mode(leaky), .leaky_shift(shift),
        .input_base_addr(in_base), .output_base_addr(out_base), .element_count(count),
        .busy(busy), .done(done), .error(error),
        .read_req_valid(read_req_valid), .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr), .read_req_len(read_req_len),
        .read_word_valid(read_word_valid), .read_word_ready(read_word_ready),
        .read_word_data(read_word_data), .read_word_keep(read_word_keep),
        .read_word_last(read_word_last),
        .write_req_valid(write_req_valid), .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr), .write_req_data(write_req_data),
        .write_req_strb(write_req_strb)
    );

    always @(posedge clk) begin
        integer beat_bytes;
        cycles <= cycles + 1;
        if (!rst_n) begin
            read_word_valid <= 1'b0;
            read_word_data <= '0;
            read_word_keep <= '0;
            read_word_last <= 1'b0;
            response_pending <= 1'b0;
            response_addr <= '0;
            response_remaining <= '0;
        end else begin
            if (read_word_valid && read_word_ready) read_word_valid <= 1'b0;
            if (!read_word_valid && response_pending) begin
                beat_bytes = response_remaining >= 4 ? 4 : response_remaining;
                read_word_data <= '0;
                read_word_keep <= (5'd1 << beat_bytes) - 1'b1;
                read_word_last <= response_remaining <= 4;
                for (int lane = 0; lane < 4; lane++) begin
                    if (lane < beat_bytes)
                        read_word_data[lane*8 +: 8] <= mem[response_addr + lane];
                end
                read_word_valid <= 1'b1;
                response_addr <= response_addr + beat_bytes;
                response_remaining <= response_remaining - beat_bytes;
                if (response_remaining <= 4) response_pending <= 1'b0;
            end
            if (read_req_valid && read_req_ready) begin
                response_pending <= 1'b1;
                response_addr <= read_req_addr;
                response_remaining <= read_req_len;
                request_count <= request_count + 1;
                if (!((request_count == 0 && read_req_len == 16) ||
                      (request_count == 1 && read_req_len == 1))) begin
                    $error("Unexpected activation request length %0d", read_req_len);
                    errors <= errors + 1;
                end
            end
            if (write_req_valid && write_req_ready) begin
                for (int lane = 0; lane < 4; lane++) begin
                    if (write_req_strb[lane])
                        mem[write_req_addr + lane] <= write_req_data[lane*8 +: 8];
                end
            end
            if (start) start_cycle <= cycles;
            if (done) done_cycle <= cycles;
        end
    end

    initial begin
        errors = 0; request_count = 0; cycles = 0; start_cycle = -1; done_cycle = -1;
        start = 0; leaky = 1; shift = 2; in_base = 0; out_base = 32; count = 17;
        read_word_valid = 0; read_word_data = 0; read_word_keep = 0; read_word_last = 0;
        response_pending = 0; response_addr = 0; response_remaining = 0;
        for (int i = 0; i < 17; i++) mem[i] = i - 9;
        repeat (3) @(posedge clk); rst_n = 1;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done); #1;
        for (int i = 0; i < 17; i++) begin
            if (mem[32+i] !== (i-9 < 0 ? ((i-9) >>> 2) : i-9)) errors++;
        end
        if (error || errors || request_count != 2)
            $fatal(1, "activation wide DDR trace failed errors=%0d requests=%0d", errors, request_count);
        $display("Parallel DDR Leaky ReLU wide golden trace passed: 17 values in 2 requests");
        $finish;
    end
endmodule
