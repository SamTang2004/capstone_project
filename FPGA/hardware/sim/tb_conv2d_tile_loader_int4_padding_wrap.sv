`timescale 1ns/1ps
module tb_conv2d_tile_loader_int4_padding_wrap;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic busy, done, error;
    logic read_req_valid, read_req_ready = 1;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_data_valid = 0, read_data_ready;
    logic [7:0] read_data = 8'h21;
    logic read_data_last = 0;
    logic cache_we, cache_matrix;
    logic [5:0] cache_addr;
    logic signed [3:0] cache_wdata;
    integer pending_responses = 0;
    integer output_count = 0;
    logic [63:0] output_seen = '0;

    always #5 clk = ~clk;

    conv2d_tile_loader_int4_ddr dut (
        .clk, .rst_n, .start, .matrix_select(1'b0),
        .input_base_addr(32'h1000), .weight_base_addr(32'h2000),
        .input_c(16'd3), .input_h(16'd64), .input_w(16'd64),
        .out_c(16'd16), .out_w(16'd64),
        .kernel_h(8'd3), .kernel_w(8'd3),
        .stride_h(8'd1), .stride_w(8'd1),
        .pad_h(8'd1), .pad_w(8'd1),
        .m0(16'd0), .k0(16'd0), .n0(16'd0),
        .tile_rows(4'd8), .tile_cols(4'd8),
        .busy, .done, .error,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_data_valid, .read_data_ready, .read_data, .read_data_last,
        .cache_we, .cache_matrix, .cache_addr, .cache_wdata
    );

    always_ff @(posedge clk) begin
        if (read_req_valid && read_req_ready) begin
            if (read_req_len != 1)
                $fatal(1, "padding-wrap loader issued len=%0d", read_req_len);
        end

        if (read_data_valid && read_data_ready) begin
            read_data_valid <= 1'b0;
            read_data_last <= 1'b0;
        end else if (!read_data_valid && pending_responses != 0) begin
            read_data_valid <= 1'b1;
            read_data_last <= 1'b1;
        end

        case ({
            read_req_valid && read_req_ready,
            read_data_valid && read_data_ready
        })
            2'b10: pending_responses <= pending_responses + 1;
            2'b01: pending_responses <= pending_responses - 1;
            default: pending_responses <= pending_responses;
        endcase

        if (cache_we) begin
            if (cache_matrix)
                $fatal(1, "padding-wrap A transaction marked as B");
            if (output_seen[cache_addr])
                $fatal(1, "padding-wrap loader duplicated cache address %0d",
                       cache_addr);
            output_seen[cache_addr] <= 1'b1;
            output_count <= output_count + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;
        wait (done);
        @(posedge clk);
        if (error)
            $fatal(1, "padding-wrap loader asserted error");
        if (output_count != 64 || output_seen != 64'hffff_ffff_ffff_ffff)
            $fatal(1, "padding-wrap outputs=%0d seen=%h",
                   output_count, output_seen);
        $display("INT4 Conv2D padding-wrap golden trace passed: outputs=%0d",
                 output_count);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "padding-wrap loader timeout");
    end
endmodule
