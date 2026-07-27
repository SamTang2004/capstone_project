`timescale 1ns/1ps

// Two-entry elastic boundary for the shared memory write port.
// Upstream ready depends only on registered occupancy, breaking long
// combinational backpressure paths through memory_io_driver.
module bcuda_write_command_fifo (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        input_valid,
    output logic        input_ready,
    input  logic [31:0] input_addr,
    input  logic [31:0] input_data,
    input  logic [3:0]  input_strb,
    input  logic        input_last,
    output logic        output_valid,
    input  logic        output_ready,
    output logic [31:0] output_addr,
    output logic [31:0] output_data,
    output logic [3:0]  output_strb,
    output logic        output_last,
    output logic [1:0]  occupancy
);
    logic [31:0] addr_mem [0:1];
    logic [31:0] data_mem [0:1];
    logic [3:0] strb_mem [0:1];
    logic last_mem [0:1];
    logic head_r, tail_r;
    logic [1:0] count_r;

    wire push = input_valid && input_ready;
    wire pop = output_valid && output_ready;

    assign input_ready = count_r != 2;
    assign output_valid = count_r != 0;
    assign output_addr = addr_mem[head_r];
    assign output_data = data_mem[head_r];
    assign output_strb = strb_mem[head_r];
    assign output_last = last_mem[head_r];
    assign occupancy = count_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            head_r <= 1'b0;
            tail_r <= 1'b0;
            count_r <= '0;
        end else begin
            if (push) begin
                addr_mem[tail_r] <= input_addr;
                data_mem[tail_r] <= input_data;
                strb_mem[tail_r] <= input_strb;
                last_mem[tail_r] <= input_last;
                tail_r <= ~tail_r;
            end
            if (pop)
                head_r <= ~head_r;

            unique case ({push, pop})
                2'b10: count_r <= count_r + 1'b1;
                2'b01: count_r <= count_r - 1'b1;
                default: count_r <= count_r;
            endcase
        end
    end
endmodule
