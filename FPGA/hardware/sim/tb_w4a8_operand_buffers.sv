module tb_w4a8_operand_buffers;
    localparam int DEPTH = 16;
    localparam int LENGTH_WIDTH = $clog2(DEPTH + 1);
    logic clk = 0;
    logic rst_n = 0;
    logic fill_start, fill_start_ready;
    logic [LENGTH_WIDTH-1:0] k_steps;
    logic activation_write_valid, activation_write_ready;
    logic [63:0] activation_write_data;
    logic weight_write_valid, weight_write_ready;
    logic [31:0] weight_write_data;
    logic fill_busy, fill_done;
    logic drain_start, drain_start_ready, tile_available;
    logic operand_valid, operand_ready;
    logic [63:0] activation_vector;
    logic [31:0] packed_weight_vector;
    logic operand_last, drain_busy, drain_done, error;
    integer errors = 0;

    always #5 clk = ~clk;

    w4a8_operand_buffers #(.DEPTH(DEPTH)) dut (.*);

    function automatic logic [63:0] activation_value(input integer tile, input integer step);
        activation_value = 64'ha500_0000_0000_0000 + tile*64'h100 + step;
    endfunction

    function automatic logic [31:0] weight_value(input integer tile, input integer step);
        weight_value = 32'h4b00_0000 + tile*32'h100 + step;
    endfunction

    task automatic write_activations(input integer tile, input integer length);
        begin
            for (int step = 0; step < length; step++) begin
                repeat ((step + tile) % 2) @(posedge clk);
                @(negedge clk);
                activation_write_data = activation_value(tile, step);
                activation_write_valid = 1;
                do @(posedge clk); while (!activation_write_ready);
                @(negedge clk);
                activation_write_valid = 0;
            end
        end
    endtask

    task automatic write_weights(input integer tile, input integer length);
        begin
            for (int step = 0; step < length; step++) begin
                repeat ((step + tile + 1) % 3) @(posedge clk);
                @(negedge clk);
                weight_write_data = weight_value(tile, step);
                weight_write_valid = 1;
                do @(posedge clk); while (!weight_write_ready);
                @(negedge clk);
                weight_write_valid = 0;
            end
        end
    endtask

    task automatic fill_tile(input integer tile, input integer length);
        begin
            wait (fill_start_ready);
            @(negedge clk);
            k_steps = length;
            fill_start = 1;
            @(negedge clk);
            fill_start = 0;
            fork
                write_activations(tile, length);
                write_weights(tile, length);
            join
            wait (!fill_busy);
        end
    endtask

    task automatic consume_tile(input integer tile, input integer length);
        begin
            wait (drain_start_ready);
            @(negedge clk);
            drain_start = 1;
            @(negedge clk);
            drain_start = 0;
            for (int step = 0; step < length; step++) begin
                operand_ready = step[0] ? 0 : 1;
                repeat (step[0] ? 2 : 0) @(posedge clk);
                operand_ready = 1;
                wait (operand_valid);
                #1;
                if (activation_vector !== activation_value(tile, step) ||
                    packed_weight_vector !== weight_value(tile, step) ||
                    operand_last !== (step == length-1)) begin
                    $error("tile=%0d step=%0d activation=%h weight=%h last=%0b",
                           tile, step, activation_vector, packed_weight_vector, operand_last);
                    errors++;
                end
                @(posedge clk);
                #1;
                operand_ready = 0;
            end
            wait (drain_done);
            @(posedge clk);
        end
    endtask

    initial begin
        #100000;
        $fatal(1, "W4A8 operand-buffer timeout");
    end

    initial begin
        fill_start = 0;
        k_steps = 0;
        activation_write_valid = 0;
        activation_write_data = 0;
        weight_write_valid = 0;
        weight_write_data = 0;
        drain_start = 0;
        operand_ready = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        fill_tile(0, 5);
        // Bank 0 drains while bank 1 receives the next tile.
        fork
            consume_tile(0, 5);
            fill_tile(1, 3);
        join
        consume_tile(1, 3);

        if (error || errors) $fatal(1, "W4A8 operand buffers failed errors=%0d protocol=%0b", errors, error);
        $display("W4A8 ping-pong operand-buffer golden trace passed");
        $finish;
    end
endmodule
