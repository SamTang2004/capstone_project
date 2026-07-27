`timescale 1ns/1ps
module tb_matrix_tile_loader_int4_ddr;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic busy, done, error;
    logic read_req_valid, read_req_ready = 1;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid = 0, read_word_ready, read_word_last = 0;
    logic [31:0] read_word_data = 0;
    logic [3:0] read_word_keep = 0;
    logic vector_valid, vector_ready = 1;
    logic vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;
    logic [7:0] memory [0:255];
    logic pending = 0;
    logic [31:0] pending_addr;
    logic [4:0] pending_len;
    integer vectors = 0;
    integer requests = 0;

    always #5 clk = ~clk;

    matrix_tile_loader_int4_ddr dut (
        .clk, .rst_n, .start, .load_array(1'b1), .load_matrix(1'b0),
        .tensor_base(32'h40), .row_stride_bytes(32'd6),
        .row_offset(16'd2), .col_offset(16'd3),
        .tile_rows(4'd3), .tile_cols(4'd5), .busy, .done, .error,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_word_valid, .read_word_ready, .read_word_data,
        .read_word_keep, .read_word_last,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data
    );

    always_ff @(posedge clk) begin
        read_word_valid <= 0;
        read_word_last <= 0;
        read_word_keep <= 0;
        if (read_req_valid && read_req_ready) begin
            pending <= 1;
            pending_addr <= read_req_addr;
            pending_len <= read_req_len;
            requests <= requests + 1;
        end
        if (pending && read_word_ready) begin
            read_word_data <= {
                pending_len > 3 ? memory[pending_addr+3] : 8'd0,
                pending_len > 2 ? memory[pending_addr+2] : 8'd0,
                pending_len > 1 ? memory[pending_addr+1] : 8'd0,
                memory[pending_addr]
            };
            read_word_keep <= pending_len >= 4 ? 4'hf :
                              pending_len == 3 ? 4'h7 :
                              pending_len == 2 ? 4'h3 : 4'h1;
            read_word_valid <= 1;
            read_word_last <= 1;
            pending <= 0;
        end
        if (vector_valid && vector_ready) begin
            if (!vector_array || vector_matrix)
                $fatal(1, "route metadata changed");
            if (vector_row !== vectors[2:0] || vector_mask !== 8'h1f)
                $fatal(1, "vector metadata row=%0d mask=%h",
                       vector_row, vector_mask);
            for (int lane = 0; lane < 5; lane++)
                if (vector_data[lane*4 +: 4] !== (lane + 3))
                    $fatal(1, "row %0d lane %0d got %h expected %h",
                           vector_row, lane, vector_data[lane*4 +: 4],
                           lane + 3);
            vectors <= vectors + 1;
        end
    end

    initial begin
        for (integer row = 0; row < 16; row++)
            for (integer byte_col = 0; byte_col < 6; byte_col++)
                memory[8'h40 + row*6 + byte_col] =
                    (((byte_col*2+1) & 8'h0f) << 4) |
                    ((byte_col*2) & 8'h0f);
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait (done);
        if (error) $fatal(1, "unexpected loader error");
        if (vectors != 3) $fatal(1, "vector count %0d != 3", vectors);
        if (requests != 3) $fatal(1, "word requests %0d != 3", requests);
        $display("INT4 vector matrix loader golden trace passed: odd offset, masks, packed words");
        $finish;
    end
endmodule
