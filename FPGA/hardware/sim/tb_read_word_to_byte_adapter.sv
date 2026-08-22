module tb_read_word_to_byte_adapter;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic word_valid, word_ready, word_last;
    logic [31:0] word_data;
    logic [3:0] word_keep;
    logic byte_valid, byte_ready, byte_last, busy;
    logic [7:0] byte_data;
    integer received;

    always #5 clk = ~clk;
    read_word_to_byte_adapter dut (.*);

    always_ff @(posedge clk) begin
        if (!rst_n) received <= 0;
        else if (byte_valid && byte_ready) begin
            if (byte_data !== received[7:0])
                $fatal(1, "adapter byte mismatch index=%0d data=%02x", received, byte_data);
            if (byte_last !== (received == 9))
                $fatal(1, "adapter last mismatch index=%0d last=%0b", received, byte_last);
            received <= received + 1;
        end
    end

    task automatic send_word(input logic [31:0] data, input logic [3:0] keep, input logic last);
        begin
            @(negedge clk);
            word_data = data; word_keep = keep; word_last = last; word_valid = 1'b1;
            do @(posedge clk); while (!word_ready);
            @(negedge clk); word_valid = 1'b0;
        end
    endtask

    initial begin
        word_valid = 0; word_data = 0; word_keep = 0; word_last = 0; byte_ready = 0;
        repeat (3) @(posedge clk); rst_n = 1;
        fork
            begin
                send_word(32'h03020100, 4'hf, 0);
                send_word(32'h07060504, 4'hf, 0);
                send_word(32'h00000908, 4'h3, 1);
            end
            begin
                repeat (5) @(posedge clk);
                byte_ready = 1;
                repeat (3) @(posedge clk); byte_ready = 0;
                repeat (2) @(posedge clk); byte_ready = 1;
            end
        join
        wait (received == 10 && !busy);
        $display("Wide-word to byte-stream adapter golden trace passed");
        $finish;
    end
endmodule
