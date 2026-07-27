`timescale 1ns/1ps
module tb_fused_conv_requant_int4_writer;
    logic clk = 0;
    logic rst_n = 0;
    logic input_valid = 0;
    logic input_ready;
    logic input_array;
    logic [5:0] input_addr;
    logic signed [31:0] input_data;
    logic input_last;
    logic write_req_valid;
    logic write_req_ready = 0;
    logic [31:0] write_req_addr, write_req_data;
    logic [3:0] write_req_strb;
    logic write_req_last;
    logic [15:0] m0 = 0, n0 = 0, out_h = 64, out_w = 64;
    logic [3:0] m_len = 8;
    logic [4:0] n_cols = 16;
    integer writes = 0;
    integer cycles = 0;

    always #5 clk = ~clk;

    fused_conv_requant_int4_writer dut (
        .clk, .rst_n, .input_valid, .input_ready, .input_array,
        .input_addr, .input_data, .input_last,
        .output_base_addr(32'h1000), .m0, .n0, .m_len, .n_cols,
        .out_h, .out_w,
        .multiplier(32'sd1), .right_shift(6'd0),
        .output_zero_point(32'sd0),
        .activation_enable(1'b1), .leaky_mode(1'b0),
        .leaky_shift(3'd0),
        .write_req_valid, .write_req_ready, .write_req_addr,
        .write_req_data, .write_req_strb, .write_req_last
    );

    function automatic logic [3:0] expected_value(
        input integer channel,
        input integer row
    );
        integer value;
        begin
            value = channel - row;
            if (value < 0)
                value = 0;
            else if (value > 7)
                value = 7;
            expected_value = value[3:0];
        end
    endfunction

    always_ff @(posedge clk) begin
        cycles <= cycles + 1;
        write_req_ready <= cycles[1:0] != 2'b01;
        if (write_req_valid && write_req_ready) begin
            if (write_req_addr != 32'h1000 + writes * 2048)
                $fatal(1, "fused address[%0d]=%h", writes,
                       write_req_addr);
            if (write_req_strb != 4'hf)
                $fatal(1, "fused strobe[%0d]=%h", writes,
                       write_req_strb);
            for (int lane = 0; lane < 8; lane++)
                if (write_req_data[lane*4 +: 4] !==
                    expected_value(writes, lane))
                    $fatal(1,
                           "fused data channel=%0d row=%0d got=%0d expected=%0d",
                           writes, lane,
                           write_req_data[lane*4 +: 4],
                           expected_value(writes, lane));
            if (write_req_last != (writes == 15))
                $fatal(1, "fused last mismatch at write %0d", writes);
            writes <= writes + 1;
        end
    end

    task automatic send_value(
        input integer channel,
        input integer row,
        input logic last
    );
        begin
            while (!input_ready) @(posedge clk);
            @(negedge clk);
            input_array = channel >= 8;
            input_addr = {row[2:0], channel[2:0]};
            input_data = channel - row;
            input_last = last;
            input_valid = 1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        // Match the engine stream: all rows/columns from array 0, followed by
        // all rows/columns from array 1.
        for (int array = 0; array < 2; array++)
            for (int row = 0; row < 8; row++)
                for (int column = 0; column < 8; column++)
                    send_value(array * 8 + column, row,
                               array == 1 && row == 7 && column == 7);
        // The controller advances its live tile outputs immediately after the
        // final stream beat. Draining must use the metadata captured above.
        m0 = 16'd8;
        n0 = 16'd16;
        m_len = 4'd3;
        n_cols = 5'd4;
        out_h = 16'd32;
        out_w = 16'd32;
        wait (writes == 16);
        $display("Fused Conv2D/requantize/ReLU packed writer golden trace passed");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
