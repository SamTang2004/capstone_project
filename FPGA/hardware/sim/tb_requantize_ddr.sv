module tb_requantize_ddr;
    logic clk = 0, rst_n = 0, start;
    logic [31:0] input_base_addr, output_base_addr, element_count;
    logic signed [31:0] multiplier, output_zero_point;
    logic [5:0] right_shift;
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
    logic [7:0] input_bytes [0:27];
    logic [7:0] output_bytes [0:7];
    logic signed [7:0] expected [0:6];
    logic response_pending;
    logic [31:0] response_addr;
    logic [4:0] response_remaining;
    integer write_count, request_count, errors;

    always #5 clk = ~clk;
    assign read_req_ready = !response_pending && !read_word_valid;
    assign write_req_ready = 1'b1;

    requantize_ddr dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .input_base_addr(input_base_addr), .output_base_addr(output_base_addr),
        .element_count(element_count), .multiplier(multiplier),
        .right_shift(right_shift), .output_zero_point(output_zero_point),
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
                        read_word_data[lane*8 +: 8] <= input_bytes[response_addr - 32'h100 + lane];
                end
                read_word_valid <= 1'b1;
                response_addr <= response_addr + beat_bytes;
                response_remaining <= response_remaining - beat_bytes;
                if (response_remaining <= 4) response_pending <= 1'b0;
            end
            if (read_req_valid && read_req_ready) begin
                if (!((request_count == 0 && read_req_addr == 32'h100 && read_req_len == 16) ||
                      (request_count == 1 && read_req_addr == 32'h110 && read_req_len == 12))) begin
                    $error("Unexpected requantize request addr=%h len=%0d", read_req_addr, read_req_len);
                    errors <= errors + 1;
                end
                request_count <= request_count + 1;
                response_pending <= 1'b1;
                response_addr <= read_req_addr;
                response_remaining <= read_req_len;
            end
            if (write_req_valid && write_req_ready) begin
                if (write_req_addr != 32'h200 + write_count * 4) errors <= errors + 1;
                if ((write_count == 0 && write_req_strb != 4'hf) ||
                    (write_count == 1 && write_req_strb != 4'h7)) errors <= errors + 1;
                for (int lane = 0; lane < 4; lane++) begin
                    if (write_req_strb[lane])
                        output_bytes[write_count*4 + lane] <= write_req_data[lane*8 +: 8];
                end
                write_count <= write_count + 1;
            end
        end
    end

    initial begin
        integer values [0:6];
        values[0] = -300; values[1] = -2; values[2] = 0; values[3] = 8;
        values[4] = 250; values[5] = 500; values[6] = -1000;
        expected[0] = -128; expected[1] = 2; expected[2] = 3; expected[3] = 7;
        expected[4] = 127; expected[5] = 127; expected[6] = -128;
        for (int i = 0; i < 7; i++) begin
            input_bytes[i*4] = values[i][7:0];
            input_bytes[i*4+1] = values[i][15:8];
            input_bytes[i*4+2] = values[i][23:16];
            input_bytes[i*4+3] = values[i][31:24];
        end
        for (int i = 0; i < 8; i++) output_bytes[i] = 0;
        write_count = 0; request_count = 0; errors = 0;
        start = 0; input_base_addr = 32'h100; output_base_addr = 32'h200;
        element_count = 7; multiplier = 1; right_shift = 1; output_zero_point = 3;
        read_word_valid = 0; read_word_data = 0; read_word_keep = 0; read_word_last = 0;
        response_pending = 0; response_addr = 0; response_remaining = 0;
        repeat (2) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done); #1;
        for (int i = 0; i < 7; i++) begin
            if ($signed(output_bytes[i]) != expected[i]) begin
                $error("Requantize value %0d expected %0d got %0d", i, expected[i], $signed(output_bytes[i]));
                errors++;
            end
        end
        if (error || write_count != 2 || request_count != 2 || errors)
            $fatal(1, "requantize wide test failed errors=%0d requests=%0d", errors, request_count);
        $display("requantize DDR wide golden trace passed: 7 values in 2 requests");
        $finish;
    end
endmodule
