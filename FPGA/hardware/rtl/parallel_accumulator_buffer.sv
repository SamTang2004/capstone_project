`timescale 1ns/1ps
module parallel_accumulator_column_bank (
    input  logic                clk,
    input  logic                write_valid,
    input  logic [2:0]          write_addr,
    input  logic signed [31:0]  write_data,
    input  logic [2:0]          read_addr,
    output logic signed [31:0]  read_data
);

    // Independent column banks turn one logical row update into eight writes.
    (* ram_style = "distributed" *)
    logic signed [31:0] mem [0:7];

    assign read_data = mem[read_addr];

    always_ff @(posedge clk) begin
        if (write_valid)
            mem[write_addr] <= write_data;
    end
endmodule

module parallel_accumulator_buffer (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 row_valid,
    output logic                 row_ready,
    input  logic                 row_clear,
    input  logic [1:0]           row_active_arrays,
    input  logic [2:0]           row_addr,
    input  logic signed [255:0]  row_data0,
    input  logic signed [255:0]  row_data1,
    output logic                 row_commit,
    output logic [2:0]           row_commit_addr,

    input  logic                 seed_valid,
    output logic                 seed_ready,
    input  logic                 seed_array,
    input  logic [5:0]           seed_addr,
    input  logic signed [31:0]   seed_data,

    input  logic                 read_req,
    input  logic                 read_array,
    input  logic [5:0]           read_addr,
    output logic                 read_valid,
    output logic signed [31:0]   read_data
);

    logic                pending_valid;
    logic [1:0]          pending_active_arrays;
    logic [2:0]          pending_row;
    logic signed [255:0] pending_data0;
    logic signed [255:0] pending_data1;

    logic [2:0]         bank_read_addr;
    logic signed [31:0] bank_read0 [0:7];
    logic signed [31:0] bank_read1 [0:7];

    assign row_ready = 1'b1;
    assign seed_ready = !pending_valid && !row_valid;
    // Accumulation and scalar draining are mutually exclusive controller states.
    assign bank_read_addr = row_valid ? row_addr : read_addr[5:3];

    generate
        for (genvar col = 0; col < 8; col++) begin : gen_accumulator_banks
            logic               write0_valid;
            logic               write1_valid;
            logic [2:0]         write0_addr;
            logic [2:0]         write1_addr;
            logic signed [31:0] write0_data;
            logic signed [31:0] write1_data;

            assign write0_valid = pending_valid ||
                (seed_valid && seed_ready && !seed_array &&
                 seed_addr[2:0] == col);
            assign write1_valid =
                (pending_valid && pending_active_arrays == 2) ||
                (seed_valid && seed_ready && seed_array &&
                 seed_addr[2:0] == col);
            assign write0_addr = pending_valid ? pending_row : seed_addr[5:3];
            assign write1_addr = pending_valid ? pending_row : seed_addr[5:3];
            assign write0_data = pending_valid ?
                pending_data0[col*32 +: 32] : seed_data;
            assign write1_data = pending_valid ?
                pending_data1[col*32 +: 32] : seed_data;

            parallel_accumulator_column_bank bank0 (
                .clk(clk),
                .write_valid(write0_valid),
                .write_addr(write0_addr),
                .write_data(write0_data),
                .read_addr(bank_read_addr),
                .read_data(bank_read0[col])
            );

            parallel_accumulator_column_bank bank1 (
                .clk(clk),
                .write_valid(write1_valid),
                .write_addr(write1_addr),
                .write_data(write1_data),
                .read_addr(bank_read_addr),
                .read_data(bank_read1[col])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pending_valid <= 1'b0;
            pending_active_arrays <= '0;
            pending_row <= '0;
            pending_data0 <= '0;
            pending_data1 <= '0;
            row_commit <= 1'b0;
            row_commit_addr <= '0;
            read_valid <= 1'b0;
            read_data <= '0;
        end else begin
            row_commit <= pending_valid;
            row_commit_addr <= pending_row;
            read_valid <= read_req;

            if (read_req)
                read_data <= read_array ?
                    bank_read1[read_addr[2:0]] :
                    bank_read0[read_addr[2:0]];

            pending_valid <= row_valid;
            if (row_valid) begin
                pending_active_arrays <= row_active_arrays;
                pending_row <= row_addr;
                for (int col = 0; col < 8; col++) begin
                    pending_data0[col*32 +: 32] <= row_clear ?
                        row_data0[col*32 +: 32] :
                        bank_read0[col] + row_data0[col*32 +: 32];
                    pending_data1[col*32 +: 32] <= row_clear ?
                        row_data1[col*32 +: 32] :
                        bank_read1[col] + row_data1[col*32 +: 32];
                end
            end
        end
    end
endmodule
