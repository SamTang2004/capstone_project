`timescale 1ns/1ps
module tb_conv2d_tile_loader_int4_ddr;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic matrix_select = 0;
    logic busy, done, error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_data_valid = 0, read_data_ready, read_data_last = 0;
    logic [7:0] read_data;
    logic cache_we, cache_matrix;
    logic [5:0] cache_addr;
    logic signed [3:0] cache_wdata;
    logic [31:0] response_addr;
    logic [31:0] response_queue [0:7];
    integer accepted_reads = 0;
    integer returned_reads = 0;
    integer max_outstanding = 0;
    integer service_delay = 80;
    integer writes, expected, row, col, logical_index;
    integer k_index, channel_index, kernel_x, input_x;
    integer response_element;
    integer cycle_count = 0;
    integer last_weight_issue_cycle = -1;
    integer max_weight_issue_gap = 0;

    always #5 clk = ~clk;
    assign read_req_ready = accepted_reads - returned_reads < 8;
    function automatic integer input_value(input integer index);
        input_value = ((index * 5 + 3) & 15) - 8;
    endfunction
    function automatic integer weight_value(input integer index);
        weight_value = ((index * 7 + 1) & 15) - 8;
    endfunction

    always_comb begin
        if (response_addr >= 32'h200) begin
            response_element = (response_addr - 32'h200) * 2;
            read_data = ((weight_value(response_element + 1) & 8'h0f) << 4) |
                        (weight_value(response_element) & 8'h0f);
        end else begin
            response_element = (response_addr - 32'h100) * 2;
            read_data = ((input_value(response_element + 1) & 8'h0f) << 4) |
                        (input_value(response_element) & 8'h0f);
        end
    end

    conv2d_tile_loader_int4_ddr dut (
        .clk, .rst_n, .start, .matrix_select,
        .input_base_addr(32'h100), .weight_base_addr(32'h200),
        .input_c(16'd2), .input_h(16'd3), .input_w(16'd5),
        .out_c(16'd5), .out_w(16'd5),
        .kernel_h(8'd1), .kernel_w(8'd3),
        .stride_h(8'd1), .stride_w(8'd1),
        .pad_h(8'd0), .pad_w(8'd1),
        .m0(16'd0), .k0(16'd1), .n0(16'd1),
        .tile_rows(matrix_select ? 4'd5 : 4'd3),
        .tile_cols(matrix_select ? 4'd4 : 4'd5),
        .busy, .done, .error,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_data_valid, .read_data_ready, .read_data, .read_data_last,
        .cache_we, .cache_matrix, .cache_addr, .cache_wdata
    );

    always_ff @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (read_req_valid && read_req_ready) begin
            if (read_req_len != 1) $fatal(1, "loader requested non-scalar response");
            if (^read_req_addr === 1'bx) $fatal(1, "loader requested unknown address");
            response_queue[accepted_reads & 7] <= read_req_addr;
            accepted_reads <= accepted_reads + 1;
            if (matrix_select) begin
                if (last_weight_issue_cycle >= 0 &&
                    cycle_count - last_weight_issue_cycle >
                    max_weight_issue_gap)
                    max_weight_issue_gap <=
                        cycle_count - last_weight_issue_cycle;
                last_weight_issue_cycle <= cycle_count;
            end
        end
        if (accepted_reads - returned_reads > max_outstanding)
            max_outstanding <= accepted_reads - returned_reads;
        if (service_delay > 0)
            service_delay <= service_delay - 1;

        if (read_data_valid && read_data_ready) begin
            read_data_valid <= 0;
            read_data_last <= 0;
        end else if (!read_data_valid && service_delay == 0 &&
                     returned_reads < accepted_reads) begin
            response_addr <= response_queue[returned_reads & 7];
            returned_reads <= returned_reads + 1;
            read_data_valid <= 1;
            read_data_last <= 1;
        end
        if (cache_we) begin
            row = cache_addr[5:3];
            col = cache_addr[2:0];
            if (!matrix_select) begin
                k_index = 1 + col;
                channel_index = k_index / 3;
                kernel_x = k_index % 3;
                input_x = row + kernel_x - 1;
                logical_index = channel_index*15 + input_x;
                expected = (input_x < 0 || input_x >= 5) ?
                           0 : input_value(logical_index);
            end else begin
                // K rows 1..5 map (ic=0,kw=1..2),(ic=1,kw=0..2).
                k_index = 1 + row;
                channel_index = k_index / 3;
                kernel_x = k_index % 3;
                logical_index = ((1 + col) * 2 + channel_index) * 3 + kernel_x;
                expected = weight_value(logical_index);
            end
            if (cache_wdata !== expected[3:0])
                $fatal(1, "matrix=%0d addr=%0d got=%0d expected=%0d",
                       matrix_select, cache_addr, cache_wdata, expected);
            if (cache_matrix != matrix_select) $fatal(1, "matrix tag mismatch");
            writes <= writes + 1;
        end
    end

    task automatic run_loader(input logic select_b, input integer expected_writes);
        begin
            matrix_select = select_b;
            writes = 0;
            if (select_b) begin
                last_weight_issue_cycle = -1;
                max_weight_issue_gap = 0;
            end
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            wait (done);
            if (error) $fatal(1, "loader error for matrix %0d", select_b);
            if (writes != expected_writes)
                $fatal(1, "matrix %0d writes=%0d expected=%0d",
                       select_b, writes, expected_writes);
            @(posedge clk);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        run_loader(1'b0, 15);
        run_loader(1'b1, 20);
        if (max_outstanding < 4)
            $fatal(1, "multi-read window did not fill: max outstanding=%0d",
                   max_outstanding);
        // The three-stage weight Horner path generates one request every
        // eight clocks. The former multiply/add state pairs required twelve.
        if (max_weight_issue_gap > 8)
            $fatal(1, "weight address cadence regressed: max gap=%0d cycles",
                   max_weight_issue_gap);
        $display("INT4 Conv2D multi-read/reorder golden trace passed: max outstanding=%0d",
                 max_outstanding);
        $display("Pipelined weight address cadence: max request gap=%0d cycles",
                 max_weight_issue_gap);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout");
    end
endmodule
