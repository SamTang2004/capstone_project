`timescale 1ns/1ps
module int4_streaming_scalar_row_packer (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               finish,
    input  logic               transaction_array,
    input  logic               transaction_matrix,
    input  logic [3:0]         expected_rows,
    input  logic [7:0]         expected_lane_mask,
    input  logic               scalar_valid,
    input  logic [5:0]         scalar_addr,
    input  logic signed [3:0]  scalar_data,
    output logic               vector_valid,
    input  logic               vector_ready,
    output logic               vector_array,
    output logic               vector_matrix,
    output logic [2:0]         vector_row,
    output logic [7:0]         vector_mask,
    output logic [31:0]        vector_data,
    output logic               done,
    output logic               error
);
    logic active_r;
    logic finish_r;
    logic array_r, matrix_r;
    logic [3:0] rows_r;
    logic [7:0] lane_mask_r;
    logic [31:0] row_data [0:7];
    logic [7:0] row_mask [0:7];
    logic [7:0] row_ready;
    logic [3:0] emit_row;

    wire scalar_in_range =
        scalar_addr[5:3] < rows_r &&
        lane_mask_r[scalar_addr[2:0]];
    wire [7:0] scalar_lane_bit = 8'b1 << scalar_addr[2:0];
    wire scalar_completes_row =
        scalar_valid && scalar_in_range &&
        ((row_mask[scalar_addr[5:3]] | scalar_lane_bit) == lane_mask_r);
    wire output_fire = vector_valid && vector_ready;

    assign vector_valid =
        active_r && emit_row < rows_r && row_ready[emit_row[2:0]];
    assign vector_array = array_r;
    assign vector_matrix = matrix_r;
    assign vector_row = emit_row[2:0];
    assign vector_mask = lane_mask_r;
    assign vector_data = row_data[emit_row[2:0]];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active_r <= 1'b0;
            finish_r <= 1'b0;
            array_r <= 1'b0;
            matrix_r <= 1'b0;
            rows_r <= '0;
            lane_mask_r <= '0;
            row_ready <= '0;
            emit_row <= '0;
            done <= 1'b0;
            error <= 1'b0;
            for (int row = 0; row < 8; row++) begin
                row_data[row] <= '0;
                row_mask[row] <= '0;
            end
        end else begin
            if (start) begin
                active_r <= 1'b1;
                finish_r <= 1'b0;
                array_r <= transaction_array;
                matrix_r <= transaction_matrix;
                rows_r <= expected_rows;
                lane_mask_r <= expected_lane_mask;
                row_ready <= '0;
                emit_row <= '0;
                done <= 1'b0;
                error <= expected_rows == 0 || expected_rows > 8 ||
                         expected_lane_mask == 0;
                for (int row = 0; row < 8; row++) begin
                    row_data[row] <= '0;
                    row_mask[row] <= '0;
                end
            end else if (active_r) begin
                if (finish)
                    finish_r <= 1'b1;

                if (scalar_valid) begin
                    if (!scalar_in_range ||
                        row_ready[scalar_addr[5:3]] ||
                        row_mask[scalar_addr[5:3]][scalar_addr[2:0]]) begin
                        error <= 1'b1;
                    end else begin
                        row_data[scalar_addr[5:3]]
                                [scalar_addr[2:0]*4 +: 4] <= scalar_data;
                        row_mask[scalar_addr[5:3]] <=
                            row_mask[scalar_addr[5:3]] | scalar_lane_bit;
                        if (scalar_completes_row)
                            row_ready[scalar_addr[5:3]] <= 1'b1;
                    end
                end

                if (output_fire) begin
                    row_ready[emit_row[2:0]] <= 1'b0;
                    if (emit_row + 1'b1 == rows_r) begin
                        emit_row <= emit_row + 1'b1;
                        if (finish_r || finish) begin
                            active_r <= 1'b0;
                            done <= 1'b1;
                        end
                    end else begin
                        emit_row <= emit_row + 1'b1;
                    end
                end

                if ((finish_r || finish) && emit_row == rows_r) begin
                    active_r <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
