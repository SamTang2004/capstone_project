`timescale 1ns/1ps
module tb_conv2d_group_memory_io_first_layer;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic loader_busy, loader_done, loader_error;
    logic loader_req_valid, loader_req_ready;
    logic [31:0] loader_req_addr;
    logic [4:0] loader_req_len;
    logic loader_data_valid, loader_data_ready, loader_data_last;
    logic [7:0] loader_data;
    logic vector_valid, vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;

    logic read_word_valid, read_word_last;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic write_req_ready;
    logic mem_cmd_valid, mem_cmd_ready, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_last;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_read_valid = 0, mem_read_ready, mem_read_last = 0;
    logic [31:0] mem_read_data = 0;
    logic memio_busy, memio_error;

    logic burst_active = 0;
    logic [31:0] burst_base;
    integer burst_beat = 0;
    integer vector_count = 0;
    integer command_count = 0;
    integer cycle_count = 0;

    always #5 clk = ~clk;
    assign mem_cmd_ready = !burst_active && !mem_read_valid;

    function automatic logic [31:0] memory_word(
        input logic [31:0] address
    );
        memory_word = {
            address[7:0] + 8'd3,
            address[7:0] + 8'd2,
            address[7:0] + 8'd1,
            address[7:0]
        };
    endfunction

    conv2d_tile_group_loader_int4_ddr loader (
        .clk, .rst_n, .start,
        .skip_weight_streams(1'b0),
        .input_base_addr(32'h0000_1000),
        .weight_base_addr(32'h0001_0000),
        .input_c(16'd3), .input_h(16'd64), .input_w(16'd64),
        .out_c(16'd16), .out_w(16'd64),
        .kernel_h(8'd3), .kernel_w(8'd3),
        .stride_h(8'd1), .stride_w(8'd1),
        .pad_h(8'd1), .pad_w(8'd1),
        .m0(16'd0), .k0(16'd0), .n0(16'd0),
        .m_len(4'd8), .k_len(4'd8),
        .n_cols(5'd16), .active_arrays(2'd2),
        .busy(loader_busy), .done(loader_done), .error(loader_error),
        .read_req_valid(loader_req_valid),
        .read_req_ready(loader_req_ready),
        .read_req_addr(loader_req_addr), .read_req_len(loader_req_len),
        .read_data_valid(loader_data_valid),
        .read_data_ready(loader_data_ready),
        .read_data(loader_data), .read_data_last(loader_data_last),
        .vector_valid, .vector_ready(1'b1),
        .vector_array, .vector_matrix, .vector_row,
        .vector_mask, .vector_data
    );

    memory_io_driver memio (
        .clk, .rst_n,
        .invalidate_cache(1'b0),
        .read_req_valid(loader_req_valid),
        .read_req_ready(loader_req_ready),
        .read_req_addr(loader_req_addr),
        .read_req_len(loader_req_len),
        .read_req_word_mode(1'b0),
        .read_data_valid(loader_data_valid),
        .read_data_ready(loader_data_ready),
        .read_data(loader_data), .read_data_last(loader_data_last),
        .read_word_valid, .read_word_ready(1'b0),
        .read_word_data, .read_word_keep, .read_word_last,
        .write_req_valid(1'b0), .write_req_ready,
        .write_req_addr(32'd0), .write_req_data(32'd0),
        .write_req_strb(4'd0), .write_req_last(1'b0),
        .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write,
        .mem_cmd_addr, .mem_cmd_len,
        .mem_cmd_wdata, .mem_cmd_wstrb,
        .mem_write_valid, .mem_write_ready(1'b1),
        .mem_write_data, .mem_write_strb, .mem_write_last,
        .mem_read_valid, .mem_read_ready,
        .mem_read_data, .mem_read_last,
        .busy(memio_busy), .error(memio_error)
    );

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (mem_cmd_valid && mem_cmd_ready) begin
            if (mem_cmd_write || mem_cmd_len != 15)
                $fatal(1, "unexpected memory command write=%0b len=%0d",
                       mem_cmd_write, mem_cmd_len);
            burst_active <= 1'b1;
            burst_base <= mem_cmd_addr;
            burst_beat <= 0;
            command_count <= command_count + 1;
        end

        if (burst_active && !mem_read_valid) begin
            mem_read_valid <= 1'b1;
            mem_read_data <= memory_word(burst_base);
            mem_read_last <= 1'b0;
        end else if (mem_read_valid && mem_read_ready) begin
            if (burst_beat == 15) begin
                mem_read_valid <= 1'b0;
                mem_read_last <= 1'b0;
                burst_active <= 1'b0;
            end else begin
                burst_beat <= burst_beat + 1;
                mem_read_data <= memory_word(
                    burst_base + ((burst_beat + 1) << 2));
                mem_read_last <= burst_beat == 14;
            end
        end

        if (vector_valid)
            vector_count <= vector_count + 1;
        if (memio_error)
            $fatal(1, "memory driver asserted error at cycle %0d",
                   cycle_count);
        if (loader_error)
            $fatal(1, "Conv2D group loader asserted error at cycle %0d",
                   cycle_count);
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait (loader_done);
        if (vector_count != 24)
            $fatal(1, "first layer tile emitted %0d vectors, expected 24",
                   vector_count);
        $display("INT4 first-layer Conv2D loader/memory-IO golden trace passed: vectors=%0d commands=%0d cycles=%0d",
                 vector_count, command_count, cycle_count);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1,
            "timeout busy=%0b memio_busy=%0b memio_error=%0b vectors=%0d commands=%0d",
            loader_busy, memio_busy, memio_error, vector_count,
            command_count);
    end
endmodule
