`timescale 1ns/1ps

// Persistent packed Conv2D weight-tile cache.
//
// One entry stores eight B0 rows and eight B1 rows. Entries survive across
// activation (M) tiles and are invalidated at the start of a new command.
module int4_conv_weight_tile_cache #(
    parameter int TILE_COUNT = 256,
    parameter int INDEX_W = $clog2(TILE_COUNT)
) (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 clear,

    input  logic [INDEX_W-1:0]   query_index,
    output logic                 query_hit,

    input  logic                 capture_start,
    input  logic [INDEX_W-1:0]   capture_index,
    input  logic                 capture_valid,
    output logic                 capture_ready,
    input  logic                 capture_array,
    input  logic                 capture_matrix,
    input  logic [2:0]           capture_row,
    input  logic [7:0]           capture_mask,
    input  logic [31:0]          capture_data,
    input  logic                 capture_finish,

    input  logic                 replay_start,
    input  logic [INDEX_W-1:0]   replay_index,
    input  logic [1:0]           replay_active_arrays,
    output logic                 replay_busy,
    output logic                 replay_done,
    output logic                 replay_error,
    output logic                 replay_valid,
    input  logic                 replay_ready,
    output logic                 replay_array,
    output logic                 replay_matrix,
    output logic [2:0]           replay_row,
    output logic [7:0]           replay_mask,
    output logic [31:0]          replay_data
);
    localparam int WORD_COUNT = TILE_COUNT * 16;
    localparam int WORD_ADDR_W = $clog2(WORD_COUNT);

    (* ram_style = "block" *) logic [31:0] weight_data_mem [0:WORD_COUNT-1];
    (* ram_style = "block" *) logic [7:0] weight_mask_mem [0:WORD_COUNT-1];
    logic [TILE_COUNT-1:0] tile_valid_r;
    logic capture_active_r;
    logic [INDEX_W-1:0] capture_index_r;

    logic [INDEX_W-1:0] replay_index_r;
    logic [4:0] replay_issue_r;
    logic [4:0] replay_total_r;

    wire capture_fire = capture_valid && capture_ready &&
                        capture_matrix;
    wire replay_fire = replay_valid && replay_ready;
    wire replay_slot_available = !replay_valid || replay_ready;
    wire [WORD_ADDR_W-1:0] capture_addr =
        {capture_index_r, capture_array, capture_row};
    wire [WORD_ADDR_W-1:0] replay_addr =
        {replay_index_r, replay_issue_r[3:0]};

    assign query_hit = tile_valid_r[query_index];
    assign capture_ready = capture_active_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tile_valid_r <= '0;
            capture_active_r <= 1'b0;
            capture_index_r <= '0;
            replay_busy <= 1'b0;
            replay_done <= 1'b0;
            replay_error <= 1'b0;
            replay_valid <= 1'b0;
            replay_index_r <= '0;
            replay_issue_r <= '0;
            replay_total_r <= '0;
            replay_array <= 1'b0;
            replay_matrix <= 1'b1;
            replay_row <= '0;
            replay_mask <= '0;
            replay_data <= '0;
        end else begin
            replay_done <= 1'b0;

            if (clear) begin
                tile_valid_r <= '0;
                capture_active_r <= 1'b0;
                replay_busy <= 1'b0;
                replay_valid <= 1'b0;
                replay_error <= 1'b0;
            end

            if (capture_start && !clear) begin
                capture_active_r <= 1'b1;
                capture_index_r <= capture_index;
                tile_valid_r[capture_index] <= 1'b0;
            end

            if (capture_fire) begin
                weight_data_mem[capture_addr] <= capture_data;
                weight_mask_mem[capture_addr] <= capture_mask;
            end

            if (capture_finish && capture_active_r) begin
                capture_active_r <= 1'b0;
                tile_valid_r[capture_index_r] <= 1'b1;
            end

            if (replay_start && !replay_busy && !clear) begin
                replay_index_r <= replay_index;
                replay_issue_r <= '0;
                replay_total_r <= replay_active_arrays == 2 ?
                                  5'd16 : 5'd8;
                replay_busy <= tile_valid_r[replay_index];
                replay_error <= !tile_valid_r[replay_index];
                replay_valid <= 1'b0;
                if (!tile_valid_r[replay_index])
                    replay_done <= 1'b1;
            end else if (replay_busy) begin
                if (replay_slot_available &&
                    replay_issue_r < replay_total_r) begin
                    replay_data <= weight_data_mem[replay_addr];
                    replay_mask <= weight_mask_mem[replay_addr];
                    replay_array <= replay_issue_r[3];
                    replay_matrix <= 1'b1;
                    replay_row <= replay_issue_r[2:0];
                    replay_valid <= 1'b1;
                    replay_issue_r <= replay_issue_r + 1'b1;
                end else if (replay_fire) begin
                    replay_valid <= 1'b0;
                end

                if (replay_fire &&
                    replay_issue_r == replay_total_r) begin
                    replay_busy <= 1'b0;
                    replay_valid <= 1'b0;
                    replay_done <= 1'b1;
                end
            end
        end
    end
endmodule
