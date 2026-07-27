`timescale 1ns/1ps
module tb_memory_io_axi_bridge_outstanding;
    logic clk = 0, rst_n = 0;
    logic cmd_valid = 0, cmd_ready, cmd_write = 0;
    logic [31:0] cmd_addr = 0, cmd_wdata = 0, write_data = 0;
    logic [3:0] cmd_len = 15, cmd_wstrb = 0, write_strb = 0;
    logic write_valid = 0, write_ready, write_last = 0;
    logic read_valid, read_last, read_ready = 1, error;
    logic [31:0] read_data;
    logic [31:0] awaddr, wdata, araddr, rdata;
    logic [7:0] awlen, arlen;
    logic [2:0] awsize, arsize;
    logic [1:0] awburst, arburst, bresp = 0, rresp = 0;
    logic awvalid, awready = 1, wlast, wvalid, wready = 1;
    logic bvalid = 0, bready, arvalid, arready = 1;
    logic rvalid = 0, rlast, rready;
    logic [31:0] accepted_addr [0:7];
    integer ar_count = 0;
    integer response_burst = 0;
    integer response_beat = 0;
    integer received_words = 0;

    always #5 clk = ~clk;

    memory_io_axi_bridge #(.READ_CMD_DEPTH(8)) dut (
        .aclk(clk), .aresetn(rst_n), .cmd_valid, .cmd_ready, .cmd_write,
        .cmd_addr, .cmd_len, .cmd_wdata, .cmd_wstrb,
        .write_valid, .write_ready, .write_data, .write_strb, .write_last,
        .read_valid, .read_last, .read_ready, .read_data, .error,
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen),
        .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen),
        .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp),
        .m_axi_rvalid(rvalid), .m_axi_rlast(rlast),
        .m_axi_rready(rready)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ar_count <= 0;
            response_burst <= 0;
            response_beat <= 0;
            rvalid <= 0;
            rlast <= 0;
        end else begin
            if (arvalid && arready) begin
                if (arlen != 15 || arsize != 2 || arburst != 1)
                    $fatal(1, "invalid AR attributes");
                accepted_addr[ar_count] <= araddr;
                ar_count <= ar_count + 1;
            end
            if (!rvalid && ar_count == 8 && response_burst < 8) begin
                response_beat <= 0;
                rdata <= accepted_addr[response_burst];
                rlast <= 0;
                rvalid <= 1;
            end else if (rvalid && rready) begin
                if (rlast) begin
                    rvalid <= 0;
                    rlast <= 0;
                    response_burst <= response_burst + 1;
                end else begin
                    response_beat <= response_beat + 1;
                    rdata <= accepted_addr[response_burst] +
                             (response_beat + 1) * 4;
                    rlast <= response_beat == 14;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (read_valid && read_ready) begin
            if (read_data !==
                (received_words / 16) * 32'h40 +
                (received_words % 16) * 4)
                $fatal(1, "read[%0d]=%h", received_words, read_data);
            if (read_last != ((received_words % 16) == 15))
                $fatal(1, "read_last mismatch at %0d", received_words);
            received_words <= received_words + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (int request = 0; request < 8; request++) begin
            @(negedge clk);
            cmd_addr = request * 32'h40;
            cmd_valid = 1;
            while (!cmd_ready) @(negedge clk);
            @(posedge clk);
        end
        @(negedge clk);
        cmd_valid = 0;
        wait (ar_count == 8);
        if (received_words != 0)
            $fatal(1, "read data arrived before eight AR requests");
        wait (received_words == 128);
        if (error)
            $fatal(1, "bridge asserted error");
        $display("AXI bridge outstanding-read golden trace passed: eight queued 64-byte bursts");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout ar_count=%0d responses=%0d words=%0d",
               ar_count, response_burst, received_words);
    end
endmodule
