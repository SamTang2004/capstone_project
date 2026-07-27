`timescale 1ns/1ps
module lookahead_tile_bram #(
    parameter int DEPTH = 32,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic              clk,
    input  logic              write_enable,
    input  logic [ADDR_W-1:0] write_address,
    input  logic [31:0]       write_data,
    input  logic [ADDR_W-1:0] read_address,
    output logic [31:0]       read_data
);
    (* ram_style = "block" *) logic [31:0] memory [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (write_enable)
            memory[write_address] <= write_data;
        read_data <= memory[read_address];
    end
endmodule

module lookahead_tile_loader_int4 #(
    parameter int TILE_BANKS = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        descriptor_valid,
    output logic        descriptor_ready,
    input  logic [15:0] descriptor_id,
    input  logic [31:0] descriptor_a_base,
    input  logic [31:0] descriptor_b0_base,
    input  logic [31:0] descriptor_b1_base,
    input  logic [15:0] descriptor_a_stride,
    input  logic [15:0] descriptor_b0_stride,
    input  logic [15:0] descriptor_b1_stride,
    input  logic [7:0]  descriptor_a_row_mask,
    input  logic [7:0]  descriptor_b_row_mask,
    input  logic [7:0]  descriptor_a_lane_mask,
    input  logic [7:0]  descriptor_b0_lane_mask,
    input  logic [7:0]  descriptor_b1_lane_mask,
    input  logic        descriptor_external_fill,

    input  logic        external_vector_valid,
    output logic        external_vector_ready,
    input  logic        external_vector_array,
    input  logic        external_vector_matrix,
    input  logic [2:0]  external_vector_row,
    input  logic [31:0] external_vector_data,
    input  logic        external_commit_valid,
    output logic        external_commit_ready,

    output logic        read_req_valid,
    input  logic        read_req_ready,
    output logic [31:0] read_req_addr,
    output logic [4:0]  read_req_len,
    input  logic        read_word_valid,
    output logic        read_word_ready,
    input  logic [31:0] read_word_data,
    input  logic [3:0]  read_word_keep,
    input  logic        read_word_last,

    output logic        output_valid,
    input  logic        output_ready,
    output logic [15:0] output_descriptor_id,
    output logic [2:0]  output_row,
    output logic [7:0]  output_a_mask,
    output logic [31:0] output_a_data,
    output logic [7:0]  output_b0_mask,
    output logic [31:0] output_b0_data,
    output logic [7:0]  output_b1_mask,
    output logic [31:0] output_b1_data,
    output logic        group_done,

    output logic [$clog2(TILE_BANKS):0] allocated_tiles,
    output logic [$clog2(TILE_BANKS):0] ready_tiles,
    output logic [7:0]  outstanding_words,
    output logic        cache_full,
    output logic        error
);

    localparam int BANK_W = $clog2(TILE_BANKS);
    localparam int ROWS_PER_TILE = 8;
    localparam int WORDS_PER_TILE = 24;
    localparam int MEM_DEPTH = TILE_BANKS * ROWS_PER_TILE;
    localparam int MEM_ADDR_W = $clog2(MEM_DEPTH);
    localparam int OUTPUT_FIFO_DEPTH = 4;

    logic [31:0] a_base_mem [0:TILE_BANKS-1];
    logic [31:0] b0_base_mem [0:TILE_BANKS-1];
    logic [31:0] b1_base_mem [0:TILE_BANKS-1];
    logic [15:0] a_stride_mem [0:TILE_BANKS-1];
    logic [15:0] b0_stride_mem [0:TILE_BANKS-1];
    logic [15:0] b1_stride_mem [0:TILE_BANKS-1];
    logic [15:0] id_mem [0:TILE_BANKS-1];
    logic [7:0] a_row_mask_mem [0:TILE_BANKS-1];
    logic [7:0] b_row_mask_mem [0:TILE_BANKS-1];
    logic [7:0] a_lane_mask_mem [0:TILE_BANKS-1];
    logic [7:0] b0_lane_mask_mem [0:TILE_BANKS-1];
    logic [7:0] b1_lane_mask_mem [0:TILE_BANKS-1];

    logic [BANK_W-1:0] allocate_bank;
    logic [BANK_W-1:0] issue_bank;
    logic [BANK_W-1:0] response_bank;
    logic [BANK_W:0] issue_pending_tiles;
    logic [BANK_W:0] schedule_pending_tiles;
    logic [BANK_W-1:0] schedule_bank;
    logic [2:0] schedule_row;
    logic stream_fill_active;
    logic [BANK_W-1:0] stream_fill_bank;

    logic issue_active;
    logic [4:0] issue_word;
    logic [31:0] issue_address;
    logic issue_subword;
    logic [4:0] response_word;
    logic [31:0] response_address;
    logic response_subword;
    logic [31:0] response_partial;
    logic response_pipe_valid;
    logic [31:0] response_pipe_data;
    logic [3:0] response_pipe_keep;
    logic response_pipe_last;
    logic [31:0] assembled_vector_data;
    logic assembled_vector_valid;

    logic [MEM_ADDR_W-1:0] data_write_address;
    logic [MEM_ADDR_W-1:0] data_read_address;
    logic a_data_write, b0_data_write, b1_data_write;
    logic [31:0] a_bram_data, b0_bram_data, b1_bram_data;

    logic fetch_valid_d;
    logic [15:0] fetch_id_d;
    logic [2:0] fetch_row_d;
    logic [7:0] fetch_a_mask_d;
    logic [7:0] fetch_b0_mask_d;
    logic [7:0] fetch_b1_mask_d;
    logic [175:0] output_fifo [0:OUTPUT_FIFO_DEPTH-1];
    logic [1:0] output_fifo_head, output_fifo_tail;
    logic [2:0] output_fifo_count;
    logic [175:0] output_fifo_data;

    wire descriptor_fire = descriptor_valid && descriptor_ready;
    wire descriptor_aligned = 1'b1;
    wire descriptor_allocate = descriptor_fire && descriptor_aligned;
    wire ddr_descriptor_allocate =
        descriptor_allocate && !descriptor_external_fill;
    wire stream_descriptor_allocate =
        descriptor_allocate && descriptor_external_fill;
    wire external_vector_fire =
        external_vector_valid && external_vector_ready;
    wire external_commit_fire =
        external_commit_valid && external_commit_ready;
    wire read_req_fire = read_req_valid && read_req_ready;
    wire read_word_fire = read_word_valid && read_word_ready;
    wire response_process = response_pipe_valid;
    wire response_finishes_tile =
        assembled_vector_valid && response_word == WORDS_PER_TILE - 1;
    wire fill_finishes_tile =
        response_finishes_tile || external_commit_fire;
    wire issue_finishes_tile =
        read_req_fire && issue_word == WORDS_PER_TILE - 1 &&
        (issue_address[1:0] == 0 || issue_subword);
    wire output_fire = output_valid && output_ready;
    wire output_finishes_tile = output_fire && output_row == 3'd7;
    wire fetch_issue =
        schedule_pending_tiles != 0 &&
        (output_fifo_count + fetch_valid_d < OUTPUT_FIFO_DEPTH ||
         output_fire);
    wire fetch_finishes_tile = fetch_issue && schedule_row == 3'd7;
    wire output_fifo_push = fetch_valid_d;

    function automatic logic [BANK_W-1:0] next_bank(
        input logic [BANK_W-1:0] bank
    );
        if (bank == TILE_BANKS - 1)
            next_bank = '0;
        else
            next_bank = bank + 1'b1;
    endfunction

    function automatic logic [31:0] mask_lanes(
        input logic [31:0] data,
        input logic [7:0] mask
    );
        logic [31:0] result;
        result = '0;
        for (int lane = 0; lane < 8; lane++)
            if (mask[lane])
                result[lane*4 +: 4] = data[lane*4 +: 4];
        return result;
    endfunction

    assign descriptor_ready =
        rst_n && allocated_tiles < TILE_BANKS &&
        (!descriptor_external_fill || !stream_fill_active);
    assign cache_full = allocated_tiles == TILE_BANKS;
    assign external_vector_ready = rst_n && stream_fill_active;
    assign external_commit_ready = rst_n && stream_fill_active;

    assign read_req_valid = issue_active;
    assign read_req_addr =
        {issue_address[31:2], 2'b00} +
        (issue_subword ? 4 : 0);
    assign read_req_len = 5'd4;
    assign read_word_ready = rst_n && outstanding_words != 0;

    assign data_write_address = external_vector_fire ?
        stream_fill_bank * ROWS_PER_TILE + external_vector_row :
        response_bank * ROWS_PER_TILE + response_word[2:0];
    assign data_read_address =
        schedule_bank * ROWS_PER_TILE + schedule_row;
    assign a_data_write =
        (assembled_vector_valid && response_word < 8) ||
        (external_vector_fire && !external_vector_matrix);
    assign b0_data_write =
        (assembled_vector_valid &&
         response_word >= 8 && response_word < 16) ||
        (external_vector_fire && external_vector_matrix &&
         !external_vector_array);
    assign b1_data_write =
        (assembled_vector_valid && response_word >= 16) ||
        (external_vector_fire && external_vector_matrix &&
         external_vector_array);
    wire [31:0] selected_write_data =
        external_vector_fire ? external_vector_data :
                               assembled_vector_data;

    always_comb begin
        assembled_vector_valid = 1'b0;
        assembled_vector_data = response_pipe_data;
        if (response_process) begin
            if (response_address[1:0] == 0) begin
                assembled_vector_valid = 1'b1;
            end else if (response_subword) begin
                assembled_vector_valid = 1'b1;
                assembled_vector_data =
                    (response_partial >>
                     (response_address[1:0] * 8)) |
                    (response_pipe_data <<
                     ((4 - response_address[1:0]) * 8));
            end
        end
    end

    lookahead_tile_bram #(
        .DEPTH(MEM_DEPTH), .ADDR_W(MEM_ADDR_W)
    ) a_storage (
        .clk,
        .write_enable(a_data_write),
        .write_address(data_write_address),
        .write_data(selected_write_data),
        .read_address(data_read_address),
        .read_data(a_bram_data)
    );

    lookahead_tile_bram #(
        .DEPTH(MEM_DEPTH), .ADDR_W(MEM_ADDR_W)
    ) b0_storage (
        .clk,
        .write_enable(b0_data_write),
        .write_address(data_write_address),
        .write_data(selected_write_data),
        .read_address(data_read_address),
        .read_data(b0_bram_data)
    );

    lookahead_tile_bram #(
        .DEPTH(MEM_DEPTH), .ADDR_W(MEM_ADDR_W)
    ) b1_storage (
        .clk,
        .write_enable(b1_data_write),
        .write_address(data_write_address),
        .write_data(selected_write_data),
        .read_address(data_read_address),
        .read_data(b1_bram_data)
    );

    assign output_fifo_data = output_fifo[output_fifo_head];
    assign output_valid = output_fifo_count != 0;
    assign output_descriptor_id = output_fifo_data[175:160];
    assign output_row = output_fifo_data[159:157];
    assign output_a_mask = output_fifo_data[156:149];
    assign output_b0_mask = output_fifo_data[148:141];
    assign output_b1_mask = output_fifo_data[140:133];
    assign output_b1_data = output_fifo_data[95:64];
    assign output_b0_data = output_fifo_data[63:32];
    assign output_a_data = output_fifo_data[31:0];

    always_ff @(posedge clk) begin : loader_state
        logic [7:0] selected_a_mask;
        logic [7:0] selected_b0_mask;
        logic [7:0] selected_b1_mask;

        if (!rst_n) begin
            allocate_bank <= '0;
            issue_bank <= '0;
            response_bank <= '0;
            schedule_bank <= '0;
            schedule_row <= '0;
            stream_fill_active <= 1'b0;
            stream_fill_bank <= '0;
            allocated_tiles <= '0;
            issue_pending_tiles <= '0;
            schedule_pending_tiles <= '0;
            ready_tiles <= '0;
            outstanding_words <= '0;
            issue_active <= 1'b0;
            issue_word <= '0;
            issue_address <= '0;
            issue_subword <= 1'b0;
            response_word <= '0;
            response_address <= '0;
            response_subword <= 1'b0;
            response_partial <= '0;
            response_pipe_valid <= 1'b0;
            response_pipe_data <= '0;
            response_pipe_keep <= '0;
            response_pipe_last <= 1'b0;
            fetch_valid_d <= 1'b0;
            fetch_id_d <= '0;
            fetch_row_d <= '0;
            fetch_a_mask_d <= '0;
            fetch_b0_mask_d <= '0;
            fetch_b1_mask_d <= '0;
            output_fifo_head <= '0;
            output_fifo_tail <= '0;
            output_fifo_count <= '0;
            group_done <= 1'b0;
            error <= 1'b0;
        end else begin
            group_done <= 1'b0;
            response_pipe_valid <= read_word_fire;
            if (read_word_fire) begin
                response_pipe_data <= read_word_data;
                response_pipe_keep <= read_word_keep;
                response_pipe_last <= read_word_last;
            end

            if (descriptor_fire && !descriptor_aligned)
                error <= 1'b1;

            if (descriptor_allocate) begin
                a_base_mem[allocate_bank] <= descriptor_a_base;
                b0_base_mem[allocate_bank] <= descriptor_b0_base;
                b1_base_mem[allocate_bank] <= descriptor_b1_base;
                a_stride_mem[allocate_bank] <= descriptor_a_stride;
                b0_stride_mem[allocate_bank] <= descriptor_b0_stride;
                b1_stride_mem[allocate_bank] <= descriptor_b1_stride;
                id_mem[allocate_bank] <= descriptor_id;
                a_row_mask_mem[allocate_bank] <= descriptor_a_row_mask;
                b_row_mask_mem[allocate_bank] <= descriptor_b_row_mask;
                a_lane_mask_mem[allocate_bank] <= descriptor_a_lane_mask;
                b0_lane_mask_mem[allocate_bank] <=
                    descriptor_b0_lane_mask;
                b1_lane_mask_mem[allocate_bank] <=
                    descriptor_b1_lane_mask;
                allocate_bank <= next_bank(allocate_bank);
            end
            if (stream_descriptor_allocate) begin
                stream_fill_active <= 1'b1;
                stream_fill_bank <= allocate_bank;
            end
            if (ddr_descriptor_allocate && allocated_tiles == 0)
                response_address <= descriptor_a_base;
            if (external_commit_fire)
                stream_fill_active <= 1'b0;

            if (!issue_active && issue_pending_tiles != 0) begin
                issue_active <= 1'b1;
                issue_word <= '0;
                issue_address <= a_base_mem[issue_bank];
                issue_subword <= 1'b0;
            end

            if (read_req_fire) begin
                if (issue_address[1:0] != 0 && !issue_subword) begin
                    issue_subword <= 1'b1;
                end else begin
                    issue_subword <= 1'b0;
                    if (issue_word == 5'd7)
                        issue_address <= b0_base_mem[issue_bank];
                    else if (issue_word == 5'd15)
                        issue_address <= b1_base_mem[issue_bank];
                    else if (issue_word < 5'd7)
                        issue_address <=
                            issue_address + a_stride_mem[issue_bank];
                    else if (issue_word < 5'd15)
                        issue_address <=
                            issue_address + b0_stride_mem[issue_bank];
                    else
                        issue_address <=
                            issue_address + b1_stride_mem[issue_bank];

                    if (issue_word == WORDS_PER_TILE - 1) begin
                        issue_active <= 1'b0;
                        issue_word <= '0;
                        issue_bank <= next_bank(issue_bank);
                    end else begin
                        issue_word <= issue_word + 1'b1;
                    end
                end
            end

            if (response_process) begin
                if (response_pipe_keep != 4'hf || !response_pipe_last)
                    error <= 1'b1;

                if (response_address[1:0] != 0 &&
                    !response_subword) begin
                    response_partial <= response_pipe_data;
                    response_subword <= 1'b1;
                end else begin
                    response_subword <= 1'b0;
                    if (response_word == 5'd7)
                        response_address <= b0_base_mem[response_bank];
                    else if (response_word == 5'd15)
                        response_address <= b1_base_mem[response_bank];
                    else if (response_word < 5'd7)
                        response_address <=
                            response_address +
                            a_stride_mem[response_bank];
                    else if (response_word < 5'd15)
                        response_address <=
                            response_address +
                            b0_stride_mem[response_bank];
                    else
                        response_address <=
                            response_address +
                            b1_stride_mem[response_bank];

                    if (response_word == WORDS_PER_TILE - 1) begin
                        response_word <= '0;
                        response_bank <= next_bank(response_bank);
                        response_address <=
                            a_base_mem[next_bank(response_bank)];
                    end else begin
                        response_word <= response_word + 1'b1;
                    end
                end
            end

            fetch_valid_d <= fetch_issue;
            if (fetch_issue) begin
                selected_a_mask =
                    a_row_mask_mem[schedule_bank][schedule_row] ?
                    a_lane_mask_mem[schedule_bank] : 8'h00;
                selected_b0_mask =
                    b_row_mask_mem[schedule_bank][schedule_row] ?
                    b0_lane_mask_mem[schedule_bank] : 8'h00;
                selected_b1_mask =
                    b_row_mask_mem[schedule_bank][schedule_row] ?
                    b1_lane_mask_mem[schedule_bank] : 8'h00;
                fetch_id_d <= id_mem[schedule_bank];
                fetch_row_d <= schedule_row;
                fetch_a_mask_d <= selected_a_mask;
                fetch_b0_mask_d <= selected_b0_mask;
                fetch_b1_mask_d <= selected_b1_mask;
                if (schedule_row == 3'd7) begin
                    schedule_row <= '0;
                    schedule_bank <= next_bank(schedule_bank);
                end else begin
                    schedule_row <= schedule_row + 1'b1;
                end
            end

            if (output_fifo_push) begin
                output_fifo[output_fifo_tail] <= {
                    fetch_id_d,
                    fetch_row_d,
                    fetch_a_mask_d,
                    fetch_b0_mask_d,
                    fetch_b1_mask_d,
                    37'b0,
                    mask_lanes(b1_bram_data, fetch_b1_mask_d),
                    mask_lanes(b0_bram_data, fetch_b0_mask_d),
                    mask_lanes(a_bram_data, fetch_a_mask_d)
                };
                output_fifo_tail <= output_fifo_tail + 1'b1;
            end
            if (output_fire)
                output_fifo_head <= output_fifo_head + 1'b1;
            if (output_finishes_tile) begin
                group_done <= 1'b1;
            end

            unique case ({output_fifo_push, output_fire})
                2'b10: output_fifo_count <= output_fifo_count + 1'b1;
                2'b01: output_fifo_count <= output_fifo_count - 1'b1;
                default: output_fifo_count <= output_fifo_count;
            endcase

            unique case ({descriptor_allocate, output_finishes_tile})
                2'b10: allocated_tiles <= allocated_tiles + 1'b1;
                2'b01: allocated_tiles <= allocated_tiles - 1'b1;
                default: allocated_tiles <= allocated_tiles;
            endcase

            unique case ({ddr_descriptor_allocate, issue_finishes_tile})
                2'b10:
                    issue_pending_tiles <= issue_pending_tiles + 1'b1;
                2'b01:
                    issue_pending_tiles <= issue_pending_tiles - 1'b1;
                default:
                    issue_pending_tiles <= issue_pending_tiles;
            endcase

            unique case ({fill_finishes_tile, output_finishes_tile})
                2'b10: ready_tiles <= ready_tiles + 1'b1;
                2'b01: ready_tiles <= ready_tiles - 1'b1;
                default: ready_tiles <= ready_tiles;
            endcase

            unique case ({fill_finishes_tile, fetch_finishes_tile})
                2'b10:
                    schedule_pending_tiles <=
                        schedule_pending_tiles + 1'b1;
                2'b01:
                    schedule_pending_tiles <=
                        schedule_pending_tiles - 1'b1;
                default:
                    schedule_pending_tiles <= schedule_pending_tiles;
            endcase

            unique case ({read_req_fire, read_word_fire})
                2'b10: outstanding_words <= outstanding_words + 1'b1;
                2'b01: outstanding_words <= outstanding_words - 1'b1;
                default: outstanding_words <= outstanding_words;
            endcase
        end
    end
endmodule
