`timescale 1ns/1ps
module tb_matrix_tile_group_loader_int4_ddr;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy, done, error;
    logic read_req_valid, read_req_ready = 1'b1;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid;
    logic read_word_ready;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep = 4'hf;
    logic read_word_last = 1'b1;
    logic vector_valid, vector_ready = 1'b1;
    logic vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;

    logic [31:0] request_queue [0:63];
    integer request_write = 0;
    integer request_read = 0;
    integer request_count = 0;
    integer vector_count = 0;
    integer cycle_count = 0;
    integer first_vector_cycle = -1;
    integer last_vector_cycle = -1;
    integer cadence_bubbles = 0;
    integer errors = 0;

    always #5 clk = ~clk;

    function automatic logic [7:0] memory_byte(input logic [31:0] address);
        memory_byte = address[7:0] ^ 8'h5a;
    endfunction

    function automatic logic [31:0] memory_word(input logic [31:0] address);
        for (int lane = 0; lane < 4; lane++)
            memory_word[lane*8 +: 8] = memory_byte(address + lane);
    endfunction

    matrix_tile_group_loader_int4_ddr dut (
        .clk, .rst_n, .start,
        .a_base(32'h0000_0100), .b_base(32'h0000_0800),
        .a_stride_bytes(32'd16), .b_stride_bytes(32'd16),
        .m0(16'd0), .n0(16'd0), .k0(16'd0),
        .m_len(4'd8), .k_len(4'd8), .n_cols(5'd16),
        .active_arrays(2'd2), .busy, .done, .error,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_word_valid, .read_word_ready, .read_word_data,
        .read_word_keep, .read_word_last,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data
    );

    assign read_word_valid =
        request_write >= 8 && request_read < request_write;
    assign read_word_data = memory_word(request_queue[request_read]);

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (read_req_valid && read_req_ready) begin
            if (read_req_addr[1:0] != 0 || read_req_len != 4) begin
                $error("group loader issued non-aligned word addr=%h len=%0d",
                       read_req_addr, read_req_len);
                errors <= errors + 1;
            end
            request_queue[request_write] <= read_req_addr;
            request_write <= request_write + 1;
            request_count <= request_count + 1;
        end

        if (read_word_valid && read_word_ready)
            request_read <= request_read + 1;

        if (vector_valid && vector_ready) begin
            logic [31:0] expected;
            logic [31:0] logical_addr;
            if (!vector_matrix)
                logical_addr = 32'h100 + vector_row * 16;
            else
                logical_addr = 32'h800 + vector_row * 16 +
                               (vector_array ? 4 : 0);
            expected = memory_word(logical_addr);
            if (vector_data !== expected || vector_mask !== 8'hff) begin
                $error("group vector mismatch matrix=%0b array=%0b row=%0d data=%h expected=%h",
                       vector_matrix, vector_array, vector_row,
                       vector_data, expected);
                errors <= errors + 1;
            end
            if (first_vector_cycle < 0)
                first_vector_cycle <= cycle_count;
            else if (cycle_count != last_vector_cycle + 1) begin
                cadence_bubbles <= cadence_bubbles +
                                   cycle_count - last_vector_cycle - 1;
            end
            last_vector_cycle <= cycle_count;
            vector_count <= vector_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;
        fork
            begin
                wait (done);
            end
            begin
                repeat (80) @(posedge clk);
                $fatal(1, "fused group loader timed out");
            end
        join_any
        disable fork;
        @(posedge clk);

        if (error || errors != 0 || request_count != 24 ||
            vector_count != 24)
            $fatal(1, "fused group loader failed errors=%0d requests=%0d vectors=%0d",
                   errors, request_count, vector_count);
        if (cadence_bubbles > 1)
            $fatal(1, "fused group loader had %0d payload bubbles",
                   cadence_bubbles);

        $display("INT4 fused group scheduler golden trace passed: 24 aligned DDR words, one A/B phase bubble, then one vector/cycle");
        $finish;
    end
endmodule
