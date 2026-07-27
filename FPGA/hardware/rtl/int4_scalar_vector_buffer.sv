`timescale 1ns/1ps
module int4_scalar_vector_buffer (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               scalar_valid,
    input  logic               scalar_array,
    input  logic               scalar_matrix,
    input  logic [5:0]         scalar_addr,
    input  logic signed [3:0]  scalar_data,
    input  logic               flush,
    output logic               flush_done,
    output logic               error,
    output logic               vector_valid,
    input  logic               vector_ready,
    output logic               vector_array,
    output logic               vector_matrix,
    output logic [2:0]         vector_row,
    output logic [7:0]         vector_mask,
    output logic [31:0]        vector_data
);
    logic [31:0] row_data [0:7];
    logic [7:0] row_mask [0:7];
    logic transaction_valid;
    logic transaction_array;
    logic transaction_matrix;
    logic draining;
    logic [2:0] drain_row;

    assign vector_valid = draining && row_mask[drain_row] != 8'd0;
    assign vector_array = transaction_array;
    assign vector_matrix = transaction_matrix;
    assign vector_row = drain_row;
    assign vector_mask = row_mask[drain_row];
    assign vector_data = row_data[drain_row];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            transaction_valid <= 1'b0;
            transaction_array <= 1'b0;
            transaction_matrix <= 1'b0;
            draining <= 1'b0;
            drain_row <= '0;
            flush_done <= 1'b0;
            error <= 1'b0;
            for (int row = 0; row < 8; row++) begin
                row_data[row] <= '0;
                row_mask[row] <= '0;
            end
        end else begin
            flush_done <= 1'b0;

            if (scalar_valid) begin
                if (draining) begin
                    error <= 1'b1;
                end else begin
                    if (!transaction_valid) begin
                        transaction_valid <= 1'b1;
                        transaction_array <= scalar_array;
                        transaction_matrix <= scalar_matrix;
                    end else if (transaction_array != scalar_array ||
                                 transaction_matrix != scalar_matrix) begin
                        error <= 1'b1;
                    end
                    row_data[scalar_addr[5:3]][scalar_addr[2:0]*4 +: 4] <= scalar_data;
                    row_mask[scalar_addr[5:3]][scalar_addr[2:0]] <= 1'b1;
                end
            end

            if (flush) begin
                if (draining) begin
                    error <= 1'b1;
                end else begin
                    draining <= 1'b1;
                    drain_row <= '0;
                end
            end else if (draining &&
                         (row_mask[drain_row] == 8'd0 || vector_ready)) begin
                row_data[drain_row] <= '0;
                row_mask[drain_row] <= '0;
                if (drain_row == 3'd7) begin
                    draining <= 1'b0;
                    transaction_valid <= 1'b0;
                    flush_done <= 1'b1;
                end else begin
                    drain_row <= drain_row + 1'b1;
                end
            end
        end
    end
endmodule
