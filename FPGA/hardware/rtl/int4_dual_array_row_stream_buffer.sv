`timescale 1ns/1ps
module int4_dual_array_row_stream_buffer (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        start_ready,
    input  logic [1:0]  active_arrays,
    input  logic [7:0]  expected_a_rows,
    input  logic [7:0]  expected_b_rows,
    input  logic        input_valid,
    output logic        input_ready,
    input  logic        input_array,
    input  logic        input_matrix,
    input  logic [2:0]  input_row,
    input  logic [7:0]  input_mask,
    input  logic [31:0] input_data,
    output logic        output_valid,
    input  logic        output_ready,
    output logic [2:0]  output_row,
    output logic [7:0]  output_a_mask,
    output logic [31:0] output_a_data,
    output logic [7:0]  output_b0_mask,
    output logic [31:0] output_b0_data,
    output logic [7:0]  output_b1_mask,
    output logic [31:0] output_b1_data,
    output logic        done,
    output logic        error,
    output logic [4:0]  customer0_occupancy,
    output logic [4:0]  customer1_occupancy
);
    logic active_r;
    logic dual_r;
    logic [7:0] a_rows_r, b_rows_r;
    logic [3:0] consume_row;

    logic [31:0] customer0_a_data [0:7];
    logic [31:0] customer0_b_data [0:7];
    logic [31:0] customer1_a_data [0:7];
    logic [31:0] customer1_b_data [0:7];
    logic [7:0] customer0_a_mask [0:7];
    logic [7:0] customer0_b_mask [0:7];
    logic [7:0] customer1_a_mask [0:7];
    logic [7:0] customer1_b_mask [0:7];
    logic [7:0] customer0_a_valid, customer0_b_valid;
    logic [7:0] customer1_a_valid, customer1_b_valid;

    wire target_valid =
        !input_matrix ? customer0_a_valid[input_row] :
        !input_array ? customer0_b_valid[input_row] :
                       customer1_b_valid[input_row];
    wire input_fire = input_valid && input_ready;
    wire a_required = a_rows_r[consume_row[2:0]];
    wire b_required = b_rows_r[consume_row[2:0]];
    wire customer0_row_ready =
        (!a_required || customer0_a_valid[consume_row[2:0]]) &&
        (!b_required || customer0_b_valid[consume_row[2:0]]);
    wire customer1_row_ready =
        !dual_r ||
        ((!a_required || customer1_a_valid[consume_row[2:0]]) &&
         (!b_required || customer1_b_valid[consume_row[2:0]]));
    wire output_fire = output_valid && output_ready;

    assign start_ready = !active_r;
    assign input_ready = active_r && !target_valid;
    assign output_valid =
        active_r && consume_row < 8 &&
        customer0_row_ready && customer1_row_ready;
    assign output_row = consume_row[2:0];
    assign output_a_mask =
        a_required ? customer0_a_mask[consume_row[2:0]] : 8'h00;
    assign output_a_data =
        a_required ? customer0_a_data[consume_row[2:0]] : 32'h0;
    assign output_b0_mask =
        b_required ? customer0_b_mask[consume_row[2:0]] : 8'h00;
    assign output_b0_data =
        b_required ? customer0_b_data[consume_row[2:0]] : 32'h0;
    assign output_b1_mask =
        dual_r && b_required ?
        customer1_b_mask[consume_row[2:0]] : 8'h00;
    assign output_b1_data =
        dual_r && b_required ?
        customer1_b_data[consume_row[2:0]] : 32'h0;

    always_comb begin
        customer0_occupancy = '0;
        customer1_occupancy = '0;
        for (int row = 0; row < 8; row++) begin
            if ((!a_rows_r[row] || customer0_a_valid[row]) &&
                (!b_rows_r[row] || customer0_b_valid[row]))
                customer0_occupancy = customer0_occupancy + 1'b1;
            if (dual_r &&
                (!a_rows_r[row] || customer1_a_valid[row]) &&
                (!b_rows_r[row] || customer1_b_valid[row]))
                customer1_occupancy = customer1_occupancy + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active_r <= 1'b0;
            dual_r <= 1'b0;
            a_rows_r <= '0;
            b_rows_r <= '0;
            consume_row <= '0;
            customer0_a_valid <= '0;
            customer0_b_valid <= '0;
            customer1_a_valid <= '0;
            customer1_b_valid <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            if (start && start_ready) begin
                active_r <= 1'b1;
                dual_r <= active_arrays == 2;
                a_rows_r <= expected_a_rows;
                b_rows_r <= expected_b_rows;
                consume_row <= '0;
                customer0_a_valid <= '0;
                customer0_b_valid <= '0;
                customer1_a_valid <= '0;
                customer1_b_valid <= '0;
                done <= 1'b0;
                error <= active_arrays == 0 || active_arrays > 2;
            end else if (start && !start_ready) begin
                error <= 1'b1;
            end

            if (input_fire) begin
                if (!input_matrix) begin
                    customer0_a_data[input_row] <= input_data;
                    customer0_a_mask[input_row] <= input_mask;
                    customer0_a_valid[input_row] <= 1'b1;
                    if (dual_r) begin
                        customer1_a_data[input_row] <= input_data;
                        customer1_a_mask[input_row] <= input_mask;
                        customer1_a_valid[input_row] <= 1'b1;
                    end
                end else if (!input_array) begin
                    customer0_b_data[input_row] <= input_data;
                    customer0_b_mask[input_row] <= input_mask;
                    customer0_b_valid[input_row] <= 1'b1;
                end else if (dual_r) begin
                    customer1_b_data[input_row] <= input_data;
                    customer1_b_mask[input_row] <= input_mask;
                    customer1_b_valid[input_row] <= 1'b1;
                end else begin
                    error <= 1'b1;
                end
            end

            if (output_fire) begin
                customer0_a_valid[consume_row[2:0]] <= 1'b0;
                customer0_b_valid[consume_row[2:0]] <= 1'b0;
                customer1_a_valid[consume_row[2:0]] <= 1'b0;
                customer1_b_valid[consume_row[2:0]] <= 1'b0;
                if (consume_row == 7) begin
                    active_r <= 1'b0;
                    consume_row <= '0;
                    done <= 1'b1;
                end else begin
                    consume_row <= consume_row + 1'b1;
                end
            end
        end
    end
endmodule
