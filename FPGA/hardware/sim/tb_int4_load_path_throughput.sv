`timescale 1ns/1ps
module tb_int4_load_path_throughput;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic loader_busy, loader_done, loader_error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid, read_word_ready, read_word_last;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic vector_valid, vector_ready = 1'b1;
    logic vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;
    logic read_data_valid, read_data_ready, read_data_last;
    logic [7:0] read_data;
    logic mem_cmd_valid, mem_cmd_ready = 1'b1, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_ready = 1'b1, mem_write_last;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_read_valid, mem_read_ready, mem_read_last;
    logic [31:0] mem_read_data;
    logic memio_busy, memio_error;
    logic [31:0] memory [0:15];
    logic [31:0] command_addr [0:31];
    integer command_write = 0;
    integer command_read = 0;
    integer cycle_count = 0;
    integer run_start_cycle;
    integer first_vector_cycle;
    integer previous_vector_cycle;
    integer run_cycles;
    integer vector_count;
    integer cold_cycles;
    integer warm_cycles;

    always #5 clk = ~clk;
    assign mem_read_valid = command_read < command_write;
    assign mem_read_data = memory[command_addr[command_read][5:2]];
    assign mem_read_last = mem_read_valid;

    memory_io_driver memio (
        .clk, .rst_n,
        .invalidate_cache(1'b0),
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_req_word_mode(1'b1),
        .read_data_valid, .read_data_ready(1'b0), .read_data, .read_data_last,
        .read_word_valid, .read_word_ready, .read_word_data,
        .read_word_keep, .read_word_last,
        .write_req_valid(1'b0), .write_req_ready(),
        .write_req_addr(32'd0), .write_req_data(32'd0),
        .write_req_strb(4'd0), .write_req_last(1'b0),
        .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write,
        .mem_cmd_addr, .mem_cmd_len, .mem_cmd_wdata, .mem_cmd_wstrb,
        .mem_write_valid, .mem_write_ready, .mem_write_data,
        .mem_write_strb, .mem_write_last,
        .mem_read_valid, .mem_read_ready, .mem_read_data, .mem_read_last,
        .busy(memio_busy), .error(memio_error)
    );

    matrix_tile_loader_int4_ddr loader (
        .clk, .rst_n, .start,
        .load_array(1'b0), .load_matrix(1'b0),
        .tensor_base(32'd0), .row_stride_bytes(32'd4),
        .row_offset(16'd0), .col_offset(16'd0),
        .tile_rows(4'd8), .tile_cols(4'd8),
        .busy(loader_busy), .done(loader_done), .error(loader_error),
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_word_valid, .read_word_ready, .read_word_data,
        .read_word_keep, .read_word_last,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data
    );

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (mem_cmd_valid && mem_cmd_ready && !mem_cmd_write) begin
            if (mem_cmd_addr[1:0] != 0 || mem_cmd_len != 0)
                $fatal(1, "unexpected AXI word addr=%h len=%0d",
                       mem_cmd_addr, mem_cmd_len);
            command_addr[command_write] <= mem_cmd_addr;
            command_write <= command_write + 1;
        end
        if (mem_read_valid && mem_read_ready)
            command_read <= command_read + 1;

        if (vector_valid && vector_ready) begin
            if (vector_mask != 8'hff || vector_data != memory[vector_row])
                $fatal(1, "vector %0d data=%h mask=%h expected=%h",
                       vector_row, vector_data, vector_mask,
                       memory[vector_row]);
            if (vector_count == 0)
                first_vector_cycle <= cycle_count;
            else if (cycle_count != previous_vector_cycle + 1)
                $fatal(1, "payload bubble before vector %0d: gap=%0d cycles",
                       vector_count, cycle_count - previous_vector_cycle);
            previous_vector_cycle <= cycle_count;
            vector_count <= vector_count + 1;
        end
    end

    task automatic run_tile(output integer elapsed_cycles);
        begin
            vector_count = 0;
            first_vector_cycle = -1;
            previous_vector_cycle = -1;
            @(negedge clk);
            run_start_cycle = cycle_count;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (loader_done);
            elapsed_cycles = cycle_count - run_start_cycle;
            if (loader_error || memio_error)
                $fatal(1, "throughput run reported an error");
            if (vector_count != 8)
                $fatal(1, "throughput vector count %0d != 8", vector_count);
        end
    endtask

    initial begin
        for (integer word = 0; word < 16; word++)
            memory[word] = 32'h7654_3210 + word * 32'h1111_1111;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_tile(cold_cycles);
        repeat (2) @(posedge clk);
        run_tile(warm_cycles);

        $display("INT4 load throughput golden trace passed: payload=8 vectors/8 cycles = 400 MB/s at 100 MHz");
        $display("INT4 direct-word throughput measured: first=%0d cycles (%0d MB/s), repeat=%0d cycles (%0d MB/s)",
                 cold_cycles, 3200/cold_cycles,
                 warm_cycles, 3200/warm_cycles);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "throughput test timeout");
    end
endmodule
