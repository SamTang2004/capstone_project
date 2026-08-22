module tb_conv2d_activation_address_gen;
    logic clk = 0, rst_n = 0, start, ready;
    logic busy, done, error, valid, pad, last;
    logic [31:0] addr;
    logic [4:0] len, row_idx, col_idx;
    logic [15:0] m0, k0;
    logic [4:0] rows, cols;
    logic [7:0] kh, kw, ph, pw;
    integer cycle, accepted, previous_accept_cycle;
    logic [31:0] got_addr [0:15];
    logic [4:0] got_len [0:15], got_row [0:15], got_col [0:15];
    logic got_pad [0:15], got_last [0:15];
    logic [31:0] held_addr;
    logic [4:0] held_len, held_row, held_col;
    logic held_pad, held_last;

    always #5 clk = ~clk;

    conv2d_activation_address_gen dut (
        .clk(clk), .rst_n(rst_n), .start(start), .input_base_addr(0),
        .input_c(16'd1), .input_h(16'd3), .input_w(16'd3), .out_w(16'd3),
        .kernel_h(kh), .kernel_w(kw), .stride_h(8'd1), .stride_w(8'd1),
        .pad_h(ph), .pad_w(pw), .m0(m0), .k0(k0),
        .tile_rows(rows), .tile_cols(cols), .busy(busy), .done(done), .error(error),
        .descriptor_valid(valid), .descriptor_ready(ready), .descriptor_addr(addr),
        .descriptor_len(len), .descriptor_pad(pad), .descriptor_row(row_idx),
        .descriptor_col(col_idx), .descriptor_last(last)
    );

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (valid && ready) begin
            if (previous_accept_cycle >= 0 && cycle != previous_accept_cycle + 2)
                $fatal(1, "JIT descriptor interval mismatch at cycle %0d after %0d", cycle, previous_accept_cycle);
            previous_accept_cycle <= cycle;
            got_addr[accepted] <= addr;
            got_len[accepted] <= len;
            got_row[accepted] <= row_idx;
            got_col[accepted] <= col_idx;
            got_pad[accepted] <= pad;
            got_last[accepted] <= last;
            accepted <= accepted + 1;
        end
    end

    task automatic launch;
        begin
            accepted = 0;
            previous_accept_cycle = -1;
            wait(!busy && !done);
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            wait(done);
            #1;
            if (error) $fatal(1, "Address generator reported an error");
        end
    endtask

    initial begin
        cycle = 0; accepted = 0; previous_accept_cycle = -1;
        start = 0; ready = 1; m0 = 0; k0 = 0; rows = 2; cols = 4;
        kh = 2; kw = 2; ph = 0; pw = 0;
        repeat (3) @(posedge clk); rst_n = 1;

        launch();
        if (accepted != 4) $fatal(1, "Expected four interior descriptors, got %0d", accepted);
        if (got_addr[0] != 0 || got_len[0] != 2 || got_row[0] != 0 || got_col[0] != 0 || got_pad[0]) $fatal(1, "Interior descriptor 0 mismatch");
        if (got_addr[1] != 3 || got_len[1] != 2 || got_row[1] != 0 || got_col[1] != 2 || got_pad[1]) $fatal(1, "Interior descriptor 1 mismatch");
        if (got_addr[2] != 1 || got_len[2] != 2 || got_row[2] != 1 || got_col[2] != 0 || got_pad[2]) $fatal(1, "Interior descriptor 2 mismatch");
        if (got_addr[3] != 4 || got_len[3] != 2 || got_row[3] != 1 || got_col[3] != 2 || got_pad[3] || !got_last[3]) $fatal(1, "Interior descriptor 3 mismatch");

        rows = 1; cols = 9; kh = 3; kw = 3; ph = 1; pw = 1;
        launch();
        if (accepted != 7) $fatal(1, "Expected seven padded descriptors, got %0d", accepted);
        if (!got_pad[0] || got_col[0] != 0 || !got_pad[1] || got_col[1] != 1 ||
            !got_pad[2] || got_col[2] != 2 || !got_pad[3] || got_col[3] != 3)
            $fatal(1, "Top/left padding descriptors mismatch");
        if (got_pad[4] || got_addr[4] != 0 || got_len[4] != 2 || got_col[4] != 4)
            $fatal(1, "First padded-window interior run mismatch");
        if (!got_pad[5] || got_col[5] != 6) $fatal(1, "Second left padding descriptor mismatch");
        if (got_pad[6] || got_addr[6] != 3 || got_len[6] != 2 || got_col[6] != 7 || !got_last[6])
            $fatal(1, "Final padded-window interior run mismatch");

        // A descriptor must remain stable for an arbitrarily delayed AXI/FIFO
        // consumer. Ordering is defined by acceptance, not by fixed latency.
        rows = 2; cols = 4; kh = 2; kw = 2; ph = 0; pw = 0;
        accepted = 0; previous_accept_cycle = -1; ready = 0;
        wait(!busy && !done);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait(valid);
        held_addr = addr; held_len = len; held_row = row_idx; held_col = col_idx;
        held_pad = pad; held_last = last;
        repeat (3) begin
            @(posedge clk); #1;
            if (!valid || addr != held_addr || len != held_len ||
                row_idx != held_row || col_idx != held_col ||
                pad != held_pad || last != held_last)
                $fatal(1, "Descriptor changed while backpressured");
        end
        @(negedge clk); ready = 1;
        wait(done); #1;
        if (error || accepted != 4)
            $fatal(1, "Backpressured descriptor run failed accepted=%0d", accepted);

        $display("Conv2D activation address JIT golden trace passed: ordered two-cycle descriptors remain stable under backpressure");
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "Address generator timeout state=%0d accepted=%0d start=%0d done=%0d",
               dut.state, accepted, start, done);
    end
endmodule
