module tb_requantize_ddr;
    logic clk = 0;
    logic rst_n = 0;
    logic start;
    logic [31:0] input_base_addr, output_base_addr, element_count;
    logic signed [31:0] multiplier, output_zero_point;
    logic [5:0] right_shift;
    logic busy, done, error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic read_data_valid, read_data_ready;
    logic [7:0] read_data;
    logic write_req_valid, write_req_ready;
    logic [31:0] write_req_addr, write_req_data;
    logic [3:0] write_req_strb;
    logic [7:0] input_bytes [0:27];
    logic [7:0] output_bytes [0:7];
    logic signed [7:0] expected [0:6];
    integer write_count;
    integer errors;

    always #5 clk = ~clk;
    assign read_req_ready = 1'b1;
    assign write_req_ready = 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            read_data_valid <= 1'b0;
            read_data <= '0;
        end else begin
            if (read_data_valid && read_data_ready) read_data_valid <= 1'b0;
            if (read_req_valid && read_req_ready) begin
                read_data <= input_bytes[read_req_addr - 32'h100];
                read_data_valid <= 1'b1;
            end
        end
    end

    requantize_ddr dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .input_base_addr(input_base_addr), .output_base_addr(output_base_addr), .element_count(element_count),
        .multiplier(multiplier), .right_shift(right_shift), .output_zero_point(output_zero_point),
        .busy(busy), .done(done), .error(error),
        .read_req_valid(read_req_valid), .read_req_ready(read_req_ready), .read_req_addr(read_req_addr),
        .read_data_valid(read_data_valid), .read_data_ready(read_data_ready), .read_data(read_data),
        .write_req_valid(write_req_valid), .write_req_ready(write_req_ready), .write_req_addr(write_req_addr),
        .write_req_data(write_req_data), .write_req_strb(write_req_strb)
    );

    always @(posedge clk) begin
        if (read_req_valid && read_req_ready && (read_req_addr < 32'h100 || read_req_addr >= 32'h11c)) begin
            $error("Unexpected requantize read address 0x%08h", read_req_addr);
            errors = errors + 1;
        end
        if (write_req_valid && write_req_ready) begin
            integer lane;
            if (write_req_addr != 32'h200 + write_count * 4) begin
                $error("Unexpected requantize write address 0x%08h", write_req_addr);
                errors = errors + 1;
            end
            if ((write_count == 0 && write_req_strb != 4'hf) || (write_count == 1 && write_req_strb != 4'h7)) begin
                $error("Unexpected requantize write strobe %h", write_req_strb);
                errors = errors + 1;
            end
            for (lane = 0; lane < 4; lane++) begin
                if (write_req_strb[lane]) output_bytes[write_count * 4 + lane] = write_req_data[lane * 8 +: 8];
            end
            write_count = write_count + 1;
        end
    end

    initial begin
        integer i;
        integer signed values [0:6];
        values[0] = -300; values[1] = -2; values[2] = 0; values[3] = 8;
        values[4] = 250; values[5] = 500; values[6] = -1000;
        expected[0] = -128; expected[1] = 2; expected[2] = 3; expected[3] = 7;
        expected[4] = 127; expected[5] = 127; expected[6] = -128;
        for (i = 0; i < 7; i++) begin
            input_bytes[i * 4] = values[i][7:0];
            input_bytes[i * 4 + 1] = values[i][15:8];
            input_bytes[i * 4 + 2] = values[i][23:16];
            input_bytes[i * 4 + 3] = values[i][31:24];
        end
        for (i = 0; i < 8; i++) output_bytes[i] = 0;
        write_count = 0; errors = 0;
        start = 0; input_base_addr = 32'h100; output_base_addr = 32'h200; element_count = 7;
        multiplier = 1; right_shift = 1; output_zero_point = 3;
        repeat (2) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done);
        if (error || write_count != 2) begin
            $error("Requantize did not complete cleanly");
            errors = errors + 1;
        end
        for (i = 0; i < 7; i++) begin
            if ($signed(output_bytes[i]) != expected[i]) begin
                $error("Requantize value %0d expected %0d got %0d", i, expected[i], $signed(output_bytes[i]));
                errors = errors + 1;
            end
        end
        if (errors != 0) $fatal(1, "requantize_ddr test failed with %0d errors", errors);
        $display("requantize DDR golden trace passed");
        $finish;
    end
endmodule
