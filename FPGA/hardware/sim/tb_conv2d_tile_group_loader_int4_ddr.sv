`timescale 1ns/1ps
module tb_conv2d_tile_group_loader_int4_ddr;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic busy, done, error;
    logic read_req_valid, read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_data_valid = 0, read_data_ready, read_data_last = 0;
    logic [7:0] read_data;
    logic vector_valid, vector_ready = 1;
    logic vector_array, vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;

    logic [31:0] response_queue [0:15];
    logic [31:0] response_addr;
    integer accepted_reads = 0;
    integer returned_reads = 0;
    integer service_delay = 80;
    integer cycle_count = 0;
    integer start_cycle = 0;
    integer vector_count = 0;
    integer max_parallel_planners = 0;
    integer first_vector_cycle = -1;
    integer first_vector_returned_reads = -1;

    always #5 clk = ~clk;
    assign read_req_ready = accepted_reads - returned_reads < 8;

    function automatic integer input_value(input integer index);
        input_value = ((index * 5 + 3) & 15) - 8;
    endfunction

    function automatic integer weight_value(input integer index);
        weight_value = ((index * 7 + 1) & 15) - 8;
    endfunction

    always_comb begin
        integer element;
        if (response_addr >= 32'h400) begin
            element = (response_addr - 32'h400) * 2;
            read_data = ((weight_value(element + 1) & 8'h0f) << 4) |
                        (weight_value(element) & 8'h0f);
        end else begin
            element = (response_addr - 32'h100) * 2;
            read_data = ((input_value(element + 1) & 8'h0f) << 4) |
                        (input_value(element) & 8'h0f);
        end
    end

    conv2d_tile_group_loader_int4_ddr dut (
        .clk, .rst_n, .start,
        .skip_weight_streams(1'b0),
        .input_base_addr(32'h100), .weight_base_addr(32'h400),
        .input_c(16'd2), .input_h(16'd3), .input_w(16'd5),
        .out_c(16'd12), .out_w(16'd5),
        .kernel_h(8'd1), .kernel_w(8'd3),
        .stride_h(8'd1), .stride_w(8'd1),
        .pad_h(8'd0), .pad_w(8'd1),
        .m0(16'd0), .k0(16'd1), .n0(16'd0),
        .m_len(4'd3), .k_len(4'd5),
        .n_cols(5'd12), .active_arrays(2'd2),
        .busy, .done, .error,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_data_valid, .read_data_ready, .read_data, .read_data_last,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data
    );

    always_ff @(posedge clk) begin
        integer active_planners;
        integer expected;
        integer k_index;
        integer channel;
        integer kernel_x;
        integer input_x;
        integer logical_index;

        cycle_count <= cycle_count + 1;
        active_planners = dut.loader_busy[0] + dut.loader_busy[1] +
                          dut.loader_busy[2];
        if (active_planners > max_parallel_planners)
            max_parallel_planners <= active_planners;

        if (read_req_valid && read_req_ready) begin
            if (read_req_len != 1)
                $fatal(1, "group loader requested a non-scalar response");
            response_queue[accepted_reads & 15] <= read_req_addr;
            accepted_reads <= accepted_reads + 1;
        end

        if (service_delay > 0)
            service_delay <= service_delay - 1;
        if (read_data_valid && read_data_ready) begin
            read_data_valid <= 0;
            read_data_last <= 0;
        end else if (!read_data_valid && service_delay == 0 &&
                     returned_reads < accepted_reads) begin
            response_addr <= response_queue[returned_reads & 15];
            returned_reads <= returned_reads + 1;
            read_data_valid <= 1;
            read_data_last <= 1;
        end

        if (vector_valid && vector_ready) begin
            if (first_vector_cycle < 0) begin
                first_vector_cycle <= cycle_count;
                first_vector_returned_reads <= returned_reads;
            end
            for (int lane = 0; lane < 8; lane++) begin
                expected = 0;
                if (vector_mask[lane]) begin
                    if (!vector_matrix) begin
                        k_index = 1 + lane;
                        channel = k_index / 3;
                        kernel_x = k_index % 3;
                        input_x = vector_row + kernel_x - 1;
                        logical_index = channel * 15 + input_x;
                        expected = (lane >= 5 || input_x < 0 || input_x >= 5) ?
                                   0 : input_value(logical_index);
                    end else begin
                        k_index = 1 + vector_row;
                        channel = k_index / 3;
                        kernel_x = k_index % 3;
                        logical_index =
                            (((vector_array ? 8 : 0) + lane) * 2 + channel) *
                            3 + kernel_x;
                        expected = weight_value(logical_index);
                    end
                    if ($signed(vector_data[lane*4 +: 4]) !== expected)
                        $fatal(1,
                            "vector matrix=%0d array=%0d row=%0d lane=%0d got=%0d expected=%0d",
                            vector_matrix, vector_array, vector_row, lane,
                            $signed(vector_data[lane*4 +: 4]), expected);
                end
            end
            vector_count <= vector_count + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start_cycle = cycle_count;
        start = 1;
        @(posedge clk);
        start = 0;
        wait (done);
        if (error)
            $fatal(1, "concurrent Conv2D group loader asserted error");
        if (vector_count != 13)
            $fatal(1, "emitted %0d vectors, expected 13", vector_count);
        if (max_parallel_planners != 3)
            $fatal(1, "only %0d Conv2D planners overlapped",
                   max_parallel_planners);
        if (first_vector_cycle < 0 || first_vector_returned_reads >= 75)
            $fatal(1,
                "row streaming did not overlap load: first-cycle=%0d returned-reads=%0d",
                first_vector_cycle, first_vector_returned_reads);
        if (cycle_count - start_cycle >= 500)
            $fatal(1, "concurrent group schedule took %0d cycles",
                   cycle_count - start_cycle);
        $display("INT4 concurrent Conv2D group golden trace passed: vectors=%0d cycles=%0d first-vector-cycle=%0d first-vector-after-reads=%0d/75 max-parallel-planners=%0d",
                 vector_count, cycle_count - start_cycle,
                 first_vector_cycle - start_cycle,
                 first_vector_returned_reads,
                 max_parallel_planners);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout");
    end
endmodule
