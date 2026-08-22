module tb_conv_writer_axi_path;
    logic clk = 0, rst_n = 0, start = 0;
    logic writer_busy, writer_done, writer_error;
    logic tile_read_req, tile_read_valid;
    logic [7:0] tile_read_addr;
    logic signed [31:0] tile_read_data;
    logic write_req_valid, write_req_last, write_req_ready;
    logic [31:0] write_req_addr;
    logic signed [31:0] write_req_data;
    logic mem_cmd_valid, mem_cmd_ready, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_ready, mem_write_last;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_busy, mem_error;
    logic [31:0] awaddr;
    logic [63:0] wdata;
    logic [7:0] wstrb;
    logic [3:0] awlen;
    logic [2:0] awsize;
    logic [1:0] awburst, bresp = 0;
    logic awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [31:0] memory [0:15];
    logic [31:0] tile [0:255];
    integer write_index, cycle_count, errors = 0;

    always #5 clk = ~clk;

    conv2d_tile_writer_ddr writer (
        .clk(clk), .rst_n(rst_n), .start(start), .conv_mode(1'b1),
        .base_addr(0), .row_stride(0), .output_base_addr(32'h100),
        .out_h(16'd2), .out_w(16'd2), .m0(16'd0), .n0(16'd0),
        .tile_rows(5'd4), .tile_cols(5'd1), .busy(writer_busy),
        .done(writer_done), .error(writer_error), .tile_read_req(tile_read_req),
        .tile_read_addr(tile_read_addr), .tile_read_valid(tile_read_valid),
        .tile_read_data(tile_read_data), .write_req_valid(write_req_valid),
        .write_req_last(write_req_last), .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr), .write_req_data(write_req_data)
    );

    memory_io_driver memio (
        .clk(clk), .rst_n(rst_n), .read_req_valid(1'b0), .read_req_ready(),
        .read_req_wide(1'b0),
        .read_req_qword(1'b0),
        .read_req_allow_ahead(1'b0),
        .read_req_addr(32'd0), .read_req_len(7'd1), .read_data_valid(),
        .read_data_ready(1'b1), .read_data(), .read_data_last(),
        .read_word_valid(), .read_word_ready(1'b0), .read_word_data(),
        .read_word_keep(), .read_word_last(),
        .read_qword_valid(), .read_qword_ready(1'b0), .read_qword_data(),
        .read_qword_keep(), .read_qword_last(),
        .write_req_valid(write_req_valid), .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr), .write_req_data(write_req_data),
        .write_req_strb(4'hf), .write_req_last(write_req_last),
        .mem_cmd_valid(mem_cmd_valid), .mem_cmd_ready(mem_cmd_ready),
        .mem_cmd_write(mem_cmd_write), .mem_cmd_addr(mem_cmd_addr),
        .mem_cmd_len(mem_cmd_len), .mem_cmd_wdata(mem_cmd_wdata),
        .mem_cmd_wstrb(mem_cmd_wstrb), .mem_write_valid(mem_write_valid),
        .mem_write_ready(mem_write_ready), .mem_write_data(mem_write_data),
        .mem_write_strb(mem_write_strb), .mem_write_last(mem_write_last),
        .mem_read_valid(1'b0), .mem_read_ready(), .mem_read_data(64'd0),
        .mem_read_last(1'b0), .busy(mem_busy), .error(mem_error),
        .cache_invalidate(1'b0), .stats_reset(1'b0)
    );

    memory_io_axi3_bridge bridge (
        .aclk(clk), .aresetn(rst_n), .cmd_valid(mem_cmd_valid),
        .cmd_ready(mem_cmd_ready), .cmd_write(mem_cmd_write),
        .cmd_addr(mem_cmd_addr), .cmd_len(mem_cmd_len),
        .cmd_wdata(mem_cmd_wdata), .cmd_wstrb(mem_cmd_wstrb),
        .write_valid(mem_write_valid), .write_ready(mem_write_ready),
        .write_data(mem_write_data), .write_strb(mem_write_strb),
        .write_last(mem_write_last), .read_valid(), .read_last(),
        .read_ready(1'b1), .read_data(), .error(),
        .m_axi_awid(), .m_axi_awaddr(awaddr), .m_axi_awlen(awlen),
        .m_axi_awsize(awsize), .m_axi_awburst(awburst), .m_axi_awlock(),
        .m_axi_awcache(), .m_axi_awprot(), .m_axi_awqos(),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wid(), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready), .m_axi_bresp(bresp),
        .m_axi_bid(6'd0), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .m_axi_arid(), .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(),
        .m_axi_arburst(), .m_axi_arlock(), .m_axi_arcache(),
        .m_axi_arprot(), .m_axi_arqos(), .m_axi_arvalid(),
        .m_axi_arready(1'b0), .m_axi_rid(6'd0), .m_axi_rdata(64'd0),
        .m_axi_rresp(2'b00), .m_axi_rvalid(1'b0), .m_axi_rlast(1'b0),
        .m_axi_rready()
    );

    // Match the accumulator's two-cycle response contract.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tile_read_valid <= 1'b0;
            tile_read_data <= '0;
        end else begin
            tile_read_valid <= tile_read_req;
            if (tile_read_req) tile_read_data <= tile[tile_read_addr];
        end
    end

    // AXI slave with deterministic, changing backpressure.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0; write_index <= 0; bvalid <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (awvalid && awready) write_index <= 0;
            if (wvalid && wready) begin
                if (wstrb[3:0] != 0)
                    memory[((awaddr - 32'h100) >> 2) + write_index*2] <= wdata[31:0];
                if (wstrb[7:4] != 0)
                    memory[((awaddr - 32'h100) >> 2) + write_index*2+1] <= wdata[63:32];
                if (wlast) bvalid <= 1'b1;
                else write_index <= write_index + 1;
            end
            if (bvalid && bready) bvalid <= 1'b0;
        end
    end
    assign awready = cycle_count[0];
    assign wready = cycle_count[1] || cycle_count[0];

    initial begin
        for (int i = 0; i < 16; i++) memory[i] = 32'h11110000 + i;
        for (int i = 0; i < 256; i++) tile[i] = 0;
        tile[0] = -4; tile[16] = -2; tile[32] = 2; tile[48] = 4;
        repeat (4) @(posedge clk); rst_n = 1;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait(writer_done);
        if (cycle_count > 80)
            $fatal(1, "Conv writer recurrence exceeded cycle budget: %0d", cycle_count);
        wait(!mem_busy); repeat (3) @(posedge clk);
        if (awlen != 1 || awsize != 3 || awburst != 1) errors++;
        if ($signed(memory[0]) != -4 || $signed(memory[1]) != -2 ||
            $signed(memory[2]) != 2 || $signed(memory[3]) != 4) begin
            $display("writer AXI values: %0d %0d %0d %0d",
                     $signed(memory[0]), $signed(memory[1]),
                     $signed(memory[2]), $signed(memory[3]));
            errors++;
        end
        if (writer_error || mem_error || errors)
            $fatal(1, "Conv writer-to-AXI path failed errors=%0d", errors);
        $display("Conv writer-to-AXI golden trace passed with two native 64-bit beats in %0d cycles", cycle_count);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk);
        $fatal(1, "Conv writer-to-AXI path timeout");
    end
endmodule
