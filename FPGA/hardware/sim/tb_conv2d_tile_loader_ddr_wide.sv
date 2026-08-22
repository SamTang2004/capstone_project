module tb_conv2d_tile_loader_ddr_wide;
    logic clk = 0, rst_n = 0, start = 0;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [6:0] read_req_len;
    logic read_word_valid, read_word_ready, read_word_last;
    logic [63:0] read_word_data;
    logic [7:0] read_word_keep;
    logic cache_we, cache_matrix, cache_word_we;
    logic [7:0] cache_addr, cache_wdata, cache_word_addr, cache_word_stride;
    logic [63:0] cache_word_data;
    logic [7:0] cache_word_keep;
    logic busy, done, error;
    logic [7:0] memory [0:63], captured [0:255];
    logic [31:0] request_addr_fifo [0:7];
    logic [6:0] request_len_fifo [0:7];
    logic [2:0] request_write_ptr, request_read_ptr;
    logic [3:0] request_count;
    integer cycles, issued, completed, max_outstanding, errors;
    integer previous_issue_cycle, max_issue_gap;

    always #5 clk = ~clk;

    conv2d_tile_loader_ddr #(.WIDE_DATA_ENABLE(1'b1)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .matrix_select(1'b0),
        .conv_mode(1'b1), .presliced_weights(1'b0),
        .base_addr(32'd0), .row_stride(32'd16),
        .input_base_addr(32'd0), .weight_base_addr(32'd0),
        .input_c(16'd1), .input_h(16'd3), .input_w(16'd3),
        .out_c(16'd1), .out_h(16'd2), .out_w(16'd2),
        .kernel_h(8'd2), .kernel_w(8'd2),
        .stride_h(8'd1), .stride_w(8'd1), .pad_h(8'd0), .pad_w(8'd0),
        .m0(16'd0), .k0(16'd0), .n0(16'd0),
        .tile_rows(5'd2), .tile_cols(5'd4),
        .busy(busy), .done(done), .error(error),
        .read_req_valid(read_req_valid), .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr), .read_req_len(read_req_len),
        .read_data_valid(1'b0), .read_data_ready(), .read_data(8'd0), .read_data_last(1'b0),
        .read_word_valid(read_word_valid), .read_word_ready(read_word_ready),
        .read_word_data(read_word_data), .read_word_keep(read_word_keep),
        .read_word_last(read_word_last),
        .cache_we(cache_we), .cache_matrix(cache_matrix),
        .cache_addr(cache_addr), .cache_wdata(cache_wdata),
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
            previous_issue_cycle <= -1;
            max_issue_gap <= 0;
        end else begin
            cycles <= cycles + 1;
            read_req_ready <= request_count < 8;
            if (push_request) begin
                request_addr_fifo[request_write_ptr] <= read_req_addr;
                request_len_fifo[request_write_ptr] <= read_req_len;
                request_write_ptr <= request_write_ptr + 1'b1;
                issued <= issued + 1;
                if (previous_issue_cycle >= 0 &&
                    cycles - previous_issue_cycle > max_issue_gap)
                    max_issue_gap <= cycles - previous_issue_cycle;
                previous_issue_cycle <= cycles;
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
            if (cache_we) captured[cache_addr] <= cache_wdata;
        end
    end

    initial begin
        errors = 0;
        for (int i = 0; i < 64; i++) memory[i] = 0;
        for (int i = 0; i < 256; i++) captured[i] = 0;
        for (int i = 0; i < 9; i++) memory[i] = i + 1;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        wait (done);
        #1;
        if (captured[0] != 1 || captured[1] != 2 ||
            captured[2] != 4 || captured[3] != 5 ||
            captured[16] != 2 || captured[17] != 3 ||
            captured[18] != 5 || captured[19] != 6) errors++;
        if (issued != 4 || completed != 4 || max_outstanding < 4) begin
            $error("lookahead coverage failed issued=%0d completed=%0d max=%0d",
                   issued, completed, max_outstanding);
            errors++;
        end
        if (max_issue_gap > 2) begin
            $error("ordered request gap exceeded two clocks: %0d", max_issue_gap);
            errors++;
        end
        if (error || errors) $fatal(1, "wide TileLoader golden trace failed errors=%0d", errors);
        $display("Wide Conv2D TileLoader golden trace passed: four requests outstanding, max issue gap=%0d clocks, eight-byte bank writes", max_issue_gap);
        $finish;
    end

    initial begin
        repeat (3000) @(posedge clk);
        $fatal(1, "wide TileLoader timeout state=%0d desc=%0d responses=%0d",
               dut.state, dut.jit_fifo_count, dut.jit_response_count);
    end
endmodule
