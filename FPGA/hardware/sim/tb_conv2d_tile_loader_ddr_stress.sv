module tb_conv2d_tile_loader_ddr_stress;
    logic clk = 0, rst_n = 0, start = 0;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [6:0] read_req_len;
    logic read_word_valid, read_word_ready, read_word_last;
    logic [63:0] read_word_data;
    logic [7:0] read_word_keep;
    logic cache_word_we;
    logic [7:0] cache_word_addr, cache_word_stride, cache_word_keep;
    logic [63:0] cache_word_data;
    logic busy, done, error;
    logic [7:0] memory [0:12287], captured [0:255];
    logic [31:0] request_addr_fifo [0:7];
    logic [6:0] request_len_fifo [0:7];
    logic [2:0] request_write_ptr, request_read_ptr;
    logic [3:0] request_count;
    integer cycles, issued, completed, max_outstanding, errors;

    always #5 clk = ~clk;

    conv2d_tile_loader_ddr #(.WIDE_DATA_ENABLE(1'b1)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .matrix_select(1'b0),
        .conv_mode(1'b1), .presliced_weights(1'b0),
        .base_addr(32'd0), .row_stride(32'd16),
        .input_base_addr(32'd0), .weight_base_addr(32'd0),
        .input_c(16'd3), .input_h(16'd64), .input_w(16'd64),
        .out_c(16'd12), .out_h(16'd64), .out_w(16'd64),
        .kernel_h(8'd3), .kernel_w(8'd3),
        .stride_h(8'd1), .stride_w(8'd1), .pad_h(8'd1), .pad_w(8'd1),
        .m0(16'd0), .k0(16'd0), .n0(16'd0),
        .tile_rows(5'd16), .tile_cols(5'd16),
        .busy(busy), .done(done), .error(error),
        .read_req_valid(read_req_valid), .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr), .read_req_len(read_req_len),
        .read_data_valid(1'b0), .read_data_ready(), .read_data(8'd0), .read_data_last(1'b0),
        .read_word_valid(read_word_valid), .read_word_ready(read_word_ready),
        .read_word_data(read_word_data), .read_word_keep(read_word_keep),
        .read_word_last(read_word_last),
        .cache_we(), .cache_matrix(), .cache_addr(), .cache_wdata(),
        .cache_word_we(cache_word_we), .cache_word_addr(cache_word_addr),
        .cache_word_stride(cache_word_stride), .cache_word_data(cache_word_data),
        .cache_word_keep(cache_word_keep),
        .profile_address_phase(), .profile_request_phase(), .profile_receive_phase()
    );

    always_ff @(posedge clk) begin
        logic push_request, launch_response;
        push_request = read_req_valid && read_req_ready;
        launch_response = (!read_word_valid || read_word_ready) && request_count != 0 && cycles > 120;
        if (!rst_n) begin
            read_req_ready <= 1'b0;
            read_word_valid <= 1'b0;
            read_word_data <= '0;
            read_word_keep <= '0;
            read_word_last <= 1'b0;
            request_write_ptr <= '0;
            request_read_ptr <= '0;
            request_count <= '0;
            cycles <= 0;
            issued <= 0;
            completed <= 0;
            max_outstanding <= 0;
        end else begin
            cycles <= cycles + 1;
            read_req_ready <= request_count < 8;
            if (push_request) begin
                request_addr_fifo[request_write_ptr] <= read_req_addr;
                request_len_fifo[request_write_ptr] <= read_req_len;
                request_write_ptr <= request_write_ptr + 1'b1;
                issued <= issued + 1;
            end
            if (read_word_valid && read_word_ready) begin
                read_word_valid <= 1'b0;
                completed <= completed + 1;
            end
            if (launch_response) begin
                read_word_data <= '0;
                for (int lane = 0; lane < 8; lane++) begin
                    if (lane < request_len_fifo[request_read_ptr])
                        read_word_data[lane*8 +: 8] <=
                            memory[request_addr_fifo[request_read_ptr] + lane];
                end
                read_word_keep <= (9'd1 << request_len_fifo[request_read_ptr]) - 1'b1;
                read_word_last <= 1'b1;
                read_word_valid <= 1'b1;
                request_read_ptr <= request_read_ptr + 1'b1;
            end
            unique case ({push_request, launch_response})
                2'b10: request_count <= request_count + 1'b1;
                2'b01: request_count <= request_count - 1'b1;
                default: begin end
            endcase
            if (issued - completed > max_outstanding)
                max_outstanding <= issued - completed;
            if (cache_word_we) begin
                for (int lane = 0; lane < 8; lane++) begin
                    if (cache_word_keep[lane])
                        captured[cache_word_addr + lane * cache_word_stride] <=
                            cache_word_data[lane*8 +: 8];
                end
            end
        end
    end

    initial begin
        errors = 0;
        for (int i = 0; i < 12288; i++) memory[i] = (i % 251) + 1;
        for (int i = 0; i < 256; i++) captured[i] = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        wait (done);
        repeat (2) @(posedge clk);
        for (int row = 0; row < 16; row++) begin
            for (int col = 0; col < 16; col++) begin
                int ic, kh, kw, ih, iw, expected;
                ic = col / 9;
                kh = (col % 9) / 3;
                kw = col % 3;
                ih = kh - 1;
                iw = row + kw - 1;
                expected = (ih < 0 || ih >= 64 || iw < 0 || iw >= 64)
                    ? 0 : memory[(ic * 64 + ih) * 64 + iw];
                if (captured[row*16 + col] != expected[7:0]) begin
                    $error("tile mismatch row=%0d col=%0d got=%0d expected=%0d",
                           row, col, captured[row*16 + col], expected);
                    errors++;
                end
            end
        end
        if (issued != completed || max_outstanding < 3) begin
            $error("queue coverage failed issued=%0d completed=%0d max=%0d",
                   issued, completed, max_outstanding);
            errors++;
        end
        if (error || errors) $fatal(1, "large Conv TileLoader stress failed errors=%0d", errors);
        $display("Large Conv TileLoader stress golden trace passed: requests=%0d max_outstanding=%0d cycles=%0d",
                 issued, max_outstanding, cycles);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk);
        $fatal(1, "large Conv TileLoader timeout state=%0d fifo=%0d responses=%0d backend=%0d",
               dut.state, dut.jit_fifo_count, dut.jit_response_count, dut.jit_backend_state);
    end
endmodule
