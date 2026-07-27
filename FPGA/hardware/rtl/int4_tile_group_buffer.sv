`timescale 1ns/1ps
module int4_tile_vector_bank (
    input  logic        clk,
    input  logic        write_valid,
    input  logic [2:0]  write_addr,
    input  logic [39:0] write_data,
    input  logic [2:0]  read_addr,
    output logic [39:0] read_data
);
    (* ram_style = "distributed" *)
    logic [39:0] mem [0:7];

    assign read_data = mem[read_addr];

    always_ff @(posedge clk) begin
        if (write_valid)
            mem[write_addr] <= write_data;
    end
endmodule

module int4_tile_group_buffer (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        group_begin_valid,
    output logic        group_begin_ready,
    input  logic        input_valid,
    output logic        input_ready,
    input  logic        input_array,
    input  logic        input_matrix,
    input  logic [2:0]  input_row,
    input  logic [7:0]  input_mask,
    input  logic [31:0] input_data,
    input  logic        group_commit_valid,
    output logic        group_commit_ready,

    output logic        output_valid,
    input  logic        output_ready,
    output logic [2:0]  output_row,
    output logic [7:0]  output_a_mask,
    output logic [31:0] output_a_data,
    output logic [7:0]  output_b0_mask,
    output logic [31:0] output_b0_data,
    output logic [7:0]  output_b1_mask,
    output logic [31:0] output_b1_data,
    output logic        group_done,
    output logic [1:0]  committed_groups
);

    localparam int VECTORS_PER_GROUP = 24;

    logic        present_mem [0:1][0:VECTORS_PER_GROUP-1];
    logic [39:0] bank_read [0:1][0:2];
    logic        fill_slot;
    logic        drain_slot;
    logic        filling;
    logic [2:0]  drain_row;
    logic        begin_fire;
    logic        commit_fire;
    logic        drain_fire;
    logic [4:0]  input_index;

    always_comb begin
        if (!input_matrix)
            input_index = {2'b00, input_row};
        else if (!input_array)
            input_index = 5'd8 + {2'b00, input_row};
        else
            input_index = 5'd16 + {2'b00, input_row};
    end

    assign group_begin_ready = !filling && committed_groups < 2;
    assign begin_fire = group_begin_valid && group_begin_ready;
    assign input_ready = filling || begin_fire;
    assign group_commit_ready = (filling || begin_fire) &&
                                committed_groups < 2;
    assign commit_fire = group_commit_valid && group_commit_ready;

    generate
        for (genvar bank_slot = 0; bank_slot < 2; bank_slot++) begin : gen_slots
            for (genvar bank_kind = 0; bank_kind < 3; bank_kind++) begin : gen_banks
                logic bank_write_valid;

                assign bank_write_valid = input_valid && input_ready &&
                    fill_slot == bank_slot &&
                    ((bank_kind == 0 && !input_matrix) ||
                     (bank_kind == 1 && input_matrix && !input_array) ||
                     (bank_kind == 2 && input_matrix && input_array));

                int4_tile_vector_bank bank (
                    .clk(clk),
                    .write_valid(bank_write_valid),
                    .write_addr(input_row),
                    .write_data({input_mask, input_data}),
                    .read_addr(drain_row),
                    .read_data(bank_read[bank_slot][bank_kind])
                );
            end
        end
    endgenerate

    assign output_valid = committed_groups != 0;
    assign output_row = drain_row;
    assign output_a_mask = 8'hff;
    assign output_b0_mask = 8'hff;
    assign output_b1_mask = 8'hff;
    always_comb begin
        output_a_data = '0;
        output_b0_data = '0;
        output_b1_data = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (present_mem[drain_slot][drain_row] &&
                bank_read[drain_slot][0][32 + lane])
                output_a_data[lane*4 +: 4] =
                    bank_read[drain_slot][0][lane*4 +: 4];
            if (present_mem[drain_slot][8 + drain_row] &&
                bank_read[drain_slot][1][32 + lane])
                output_b0_data[lane*4 +: 4] =
                    bank_read[drain_slot][1][lane*4 +: 4];
            if (present_mem[drain_slot][16 + drain_row] &&
                bank_read[drain_slot][2][32 + lane])
                output_b1_data[lane*4 +: 4] =
                    bank_read[drain_slot][2][lane*4 +: 4];
        end
    end
    assign drain_fire = output_valid && output_ready;
    assign group_done = drain_fire && drain_row == 7;

    integer slot;
    integer vector;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            fill_slot <= 1'b0;
            drain_slot <= 1'b0;
            filling <= 1'b0;
            drain_row <= '0;
            committed_groups <= '0;
            for (slot = 0; slot < 2; slot = slot + 1)
                for (vector = 0; vector < VECTORS_PER_GROUP; vector = vector + 1)
                    present_mem[slot][vector] <= 1'b0;
        end else begin
            if (begin_fire) begin
                filling <= 1'b1;
                for (vector = 0; vector < VECTORS_PER_GROUP; vector = vector + 1)
                    present_mem[fill_slot][vector] <= 1'b0;
            end

            if (input_valid && input_ready) begin
                present_mem[fill_slot][input_index] <= 1'b1;
            end

            if (commit_fire) begin
                filling <= 1'b0;
                fill_slot <= ~fill_slot;
            end

            if (drain_fire) begin
                if (drain_row == 7) begin
                    drain_row <= '0;
                    drain_slot <= ~drain_slot;
                end else begin
                    drain_row <= drain_row + 1'b1;
                end
            end

            unique case ({commit_fire, group_done})
                2'b10: committed_groups <= committed_groups + 1'b1;
                2'b01: committed_groups <= committed_groups - 1'b1;
                default: committed_groups <= committed_groups;
            endcase
        end
    end
endmodule
