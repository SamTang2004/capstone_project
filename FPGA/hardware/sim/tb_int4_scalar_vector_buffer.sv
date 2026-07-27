`timescale 1ns/1ps
module tb_int4_scalar_vector_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic scalar_valid = 1'b0;
    logic scalar_array = 1'b0;
    logic scalar_matrix = 1'b0;
    logic [5:0] scalar_addr = '0;
    logic signed [3:0] scalar_data = '0;
    logic flush = 1'b0;
    logic flush_done;
    logic error;
    logic vector_valid;
    logic vector_ready = 1'b1;
    logic vector_array;
    logic vector_matrix;
    logic [2:0] vector_row;
    logic [7:0] vector_mask;
    logic [31:0] vector_data;
    integer vectors = 0;

    always #5 clk = ~clk;

    int4_scalar_vector_buffer dut (
        .clk, .rst_n, .scalar_valid, .scalar_array, .scalar_matrix,
        .scalar_addr, .scalar_data, .flush, .flush_done, .error,
        .vector_valid, .vector_ready, .vector_array, .vector_matrix,
        .vector_row, .vector_mask, .vector_data
    );

    always @(posedge clk) begin
        if (vector_valid && vector_ready) begin
            if (vector_array !== 1'b1 || vector_matrix !== 1'b1)
                $fatal(1, "vector metadata mismatch");
            case (vector_row)
                3'd0: begin
                    if (vector_mask !== 8'hff || vector_data !== 32'h76543210)
                        $fatal(1, "full row mismatch mask=%h data=%h",
                               vector_mask, vector_data);
                end
                3'd3: begin
                    if (vector_mask !== 8'h13 || vector_data !== 32'h00040010)
                        $fatal(1, "tail row mismatch mask=%h data=%h",
                               vector_mask, vector_data);
                end
                default: $fatal(1, "unexpected vector row %0d", vector_row);
            endcase
            vectors = vectors + 1;
        end
    end

    task automatic put(input integer addr, input integer value);
        begin
            @(negedge clk);
            scalar_valid = 1'b1;
            scalar_addr = addr[5:0];
            scalar_data = value;
            @(negedge clk);
            scalar_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        scalar_array = 1'b1;
        scalar_matrix = 1'b1;

        for (int lane = 0; lane < 8; lane++)
            put(lane, lane);
        put(24, 0);
        put(25, 1);
        put(28, 4);

        @(negedge clk);
        flush = 1'b1;
        @(negedge clk);
        flush = 1'b0;

        wait (flush_done);
        @(posedge clk);
        if (error) $fatal(1, "unexpected packer error");
        if (vectors != 2) $fatal(1, "vector count %0d != 2", vectors);
        $display("INT4 scalar/vector buffer golden trace passed: full row and masked tail");
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "timeout");
    end
endmodule
