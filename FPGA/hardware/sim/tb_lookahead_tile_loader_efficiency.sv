`timescale 1ns/1ps
module tb_lookahead_tile_loader_efficiency;
    localparam int TILE_COUNT = 32;
    localparam int ROWS_PER_TILE = 8;
    localparam int WORDS_PER_TILE = 24;
    localparam int TOTAL_ROWS = TILE_COUNT * ROWS_PER_TILE;
    localparam int TOTAL_WORDS = TILE_COUNT * WORDS_PER_TILE;
    localparam int PROFILE_DIRECT_IDEAL = 0;
    localparam int PROFILE_DIRECT_CONSTRAINED = 1;
    localparam int PROFILE_EXTERNAL_IDEAL = 2;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    integer profile = PROFILE_DIRECT_IDEAL;
    integer response_latency = 2;
    integer cycle_count = 0;

    logic descriptor_valid;
    logic descriptor_ready;
    logic [15:0] descriptor_id;
    logic [31:0] descriptor_a_base;
    logic [31:0] descriptor_b0_base;
    logic [31:0] descriptor_b1_base;
    logic [15:0] descriptor_a_stride;
    logic [15:0] descriptor_b0_stride;
    logic [15:0] descriptor_b1_stride;
    logic [7:0] descriptor_a_row_mask;
    logic [7:0] descriptor_b_row_mask;
    logic [7:0] descriptor_a_lane_mask;
    logic [7:0] descriptor_b0_lane_mask;
    logic [7:0] descriptor_b1_lane_mask;
    logic descriptor_external_fill;

    logic external_vector_valid;
    logic external_vector_ready;
    logic external_vector_array;
    logic external_vector_matrix;
    logic [2:0] external_vector_row;
    logic [31:0] external_vector_data;
    logic external_commit_valid;
    logic external_commit_ready;

    logic read_req_valid;
    logic read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid;
    logic read_word_ready;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep;
    logic read_word_last;

    logic output_valid;
    logic output_ready;
    logic [15:0] output_descriptor_id;
    logic [2:0] output_row;
    logic [7:0] output_a_mask;
    logic [31:0] output_a_data;
    logic [7:0] output_b0_mask;
    logic [31:0] output_b0_data;
    logic [7:0] output_b1_mask;
    logic [31:0] output_b1_data;
    logic group_done;
    logic [2:0] allocated_tiles;
    logic [2:0] ready_tiles;
    logic [7:0] outstanding_words;
    logic cache_full;
    logic error;

    logic [31:0] response_address [0:2047];
    integer response_due [0:2047];
    integer response_write = 0;
    integer response_read = 0;
    integer descriptor_count = 0;
    integer output_count = 0;
    integer done_count = 0;
    integer external_word = 0;
    logic external_active = 1'b0;

    integer start_cycle = -1;
    integer first_request_cycle = -1;
    integer last_response_cycle = -1;
    integer first_output_cycle = -1;
    integer last_output_cycle = -1;
    integer first_done_cycle = -1;
    integer previous_done_cycle = -1;
    integer max_done_gap = 0;
    integer request_count = 0;
    integer response_count = 0;
    integer request_stall_cycles = 0;
    integer response_idle_cycles = 0;
    integer descriptor_stall_cycles = 0;
    integer output_bubble_cycles = 0;
    integer ready_empty_cycles = 0;
    integer max_ready_tiles = 0;

    always #5 clk = ~clk;

    function automatic logic [31:0] memory_word(
        input logic [31:0] address
    );
        memory_word = address ^ 32'h5a17_c3e9;
    endfunction

    assign descriptor_valid = descriptor_count < TILE_COUNT;
    assign descriptor_id = 16'h100 + descriptor_count;
    assign descriptor_a_base = 32'h1000 + descriptor_count * 32'h100;
    assign descriptor_b0_base = descriptor_a_base + 32'h40;
    assign descriptor_b1_base = descriptor_a_base + 32'h80;
    assign descriptor_a_stride = 16'd4;
    assign descriptor_b0_stride = 16'd4;
    assign descriptor_b1_stride = 16'd4;
    assign descriptor_a_row_mask = 8'hff;
    assign descriptor_b_row_mask = 8'hff;
    assign descriptor_a_lane_mask = 8'hff;
    assign descriptor_b0_lane_mask = 8'hff;
    assign descriptor_b1_lane_mask = 8'hff;
    assign descriptor_external_fill = profile == PROFILE_EXTERNAL_IDEAL;

    assign read_req_ready = rst_n &&
        (profile != PROFILE_DIRECT_CONSTRAINED || cycle_count % 8 != 3);
    assign read_word_valid =
        response_read < response_write &&
        cycle_count >= response_due[response_read] &&
        (profile != PROFILE_DIRECT_CONSTRAINED || cycle_count % 8 != 6);
    assign read_word_data = memory_word(response_address[response_read]);
    assign read_word_keep = 4'hf;
    assign read_word_last = 1'b1;

    assign external_vector_valid =
        profile == PROFILE_EXTERNAL_IDEAL &&
        external_active && external_word < WORDS_PER_TILE;
    assign external_vector_matrix = external_word >= 8;
    assign external_vector_array = external_word >= 16;
    assign external_vector_row = external_word[2:0];
    assign external_vector_data = {
        descriptor_count[7:0], external_word[7:0], 16'h4eed
    };
    assign external_commit_valid =
        profile == PROFILE_EXTERNAL_IDEAL &&
        external_active && external_word == WORDS_PER_TILE;
    assign output_ready = rst_n;

    lookahead_tile_loader_int4 #(
        .TILE_BANKS(4)
    ) dut (
        .clk, .rst_n,
        .descriptor_valid, .descriptor_ready, .descriptor_id,
        .descriptor_a_base, .descriptor_b0_base, .descriptor_b1_base,
        .descriptor_a_stride, .descriptor_b0_stride,
        .descriptor_b1_stride, .descriptor_a_row_mask,
        .descriptor_b_row_mask, .descriptor_a_lane_mask,
        .descriptor_b0_lane_mask, .descriptor_b1_lane_mask,
        .descriptor_external_fill,
        .external_vector_valid, .external_vector_ready,
        .external_vector_array, .external_vector_matrix,
        .external_vector_row, .external_vector_data,
        .external_commit_valid, .external_commit_ready,
        .read_req_valid, .read_req_ready, .read_req_addr, .read_req_len,
        .read_word_valid, .read_word_ready, .read_word_data,
        .read_word_keep, .read_word_last,
        .output_valid, .output_ready, .output_descriptor_id, .output_row,
        .output_a_mask, .output_a_data, .output_b0_mask, .output_b0_data,
        .output_b1_mask, .output_b1_data, .group_done,
        .allocated_tiles, .ready_tiles, .outstanding_words,
        .cache_full, .error
    );

    always_ff @(posedge clk) begin
        integer done_gap;
        cycle_count <= cycle_count + 1;
        if (!rst_n) begin
            response_write <= 0;
            response_read <= 0;
            descriptor_count <= 0;
            output_count <= 0;
            done_count <= 0;
            external_word <= 0;
            external_active <= 1'b0;
        end else begin
            if (descriptor_valid && descriptor_ready) begin
                if (start_cycle < 0)
                    start_cycle <= cycle_count;
                descriptor_count <= descriptor_count + 1;
                if (profile == PROFILE_EXTERNAL_IDEAL) begin
                    external_active <= 1'b1;
                    external_word <= 0;
                end
            end
            if (descriptor_valid && !descriptor_ready)
                descriptor_stall_cycles <= descriptor_stall_cycles + 1;

            if (external_vector_valid && external_vector_ready)
                external_word <= external_word + 1;
            if (external_commit_valid && external_commit_ready) begin
                external_active <= 1'b0;
                external_word <= 0;
            end

            if (read_req_valid && read_req_ready) begin
                if (read_req_len != 5'd4 || read_req_addr[1:0] != 0)
                    $fatal(1, "invalid direct request addr=%h len=%0d",
                           read_req_addr, read_req_len);
                response_address[response_write] <= read_req_addr;
                response_due[response_write] <=
                    cycle_count + response_latency;
                response_write <= response_write + 1;
                request_count <= request_count + 1;
                if (first_request_cycle < 0)
                    first_request_cycle <= cycle_count;
            end
            if (read_req_valid && !read_req_ready)
                request_stall_cycles <= request_stall_cycles + 1;
            if (read_word_valid && read_word_ready) begin
                response_read <= response_read + 1;
                response_count <= response_count + 1;
                last_response_cycle <= cycle_count;
            end
            if (first_request_cycle >= 0 &&
                response_count < TOTAL_WORDS &&
                !(read_word_valid && read_word_ready))
                response_idle_cycles <= response_idle_cycles + 1;

            if (output_valid && output_ready) begin
                if (output_descriptor_id != 16'h100 + output_count / 8 ||
                    output_row != output_count % 8)
                    $fatal(1,
                        "output order mismatch index=%0d id=%h row=%0d",
                        output_count, output_descriptor_id, output_row);
                if (output_a_mask != 8'hff ||
                    output_b0_mask != 8'hff ||
                    output_b1_mask != 8'hff)
                    $fatal(1, "unexpected output mask at row %0d",
                           output_count);
                if (first_output_cycle < 0)
                    first_output_cycle <= cycle_count;
                last_output_cycle <= cycle_count;
                output_count <= output_count + 1;
            end
            if (first_output_cycle >= 0 && output_count < TOTAL_ROWS &&
                !(output_valid && output_ready))
                output_bubble_cycles <= output_bubble_cycles + 1;
            if (first_output_cycle >= 0 && output_count < TOTAL_ROWS &&
                ready_tiles == 0)
                ready_empty_cycles <= ready_empty_cycles + 1;
            if (ready_tiles > max_ready_tiles)
                max_ready_tiles <= ready_tiles;

            if (group_done) begin
                if (first_done_cycle < 0)
                    first_done_cycle <= cycle_count;
                if (previous_done_cycle >= 0) begin
                    done_gap = cycle_count - previous_done_cycle;
                    if (done_gap > max_done_gap)
                        max_done_gap <= done_gap;
                end
                previous_done_cycle <= cycle_count;
                done_count <= done_count + 1;
            end
        end
    end

    task automatic clear_metrics;
        begin
            start_cycle = -1;
            first_request_cycle = -1;
            last_response_cycle = -1;
            first_output_cycle = -1;
            last_output_cycle = -1;
            first_done_cycle = -1;
            previous_done_cycle = -1;
            max_done_gap = 0;
            request_count = 0;
            response_count = 0;
            request_stall_cycles = 0;
            response_idle_cycles = 0;
            descriptor_stall_cycles = 0;
            output_bubble_cycles = 0;
            ready_empty_cycles = 0;
            max_ready_tiles = 0;
        end
    endtask

    task automatic run_profile(
        input integer selected_profile,
        input integer selected_latency,
        input string profile_name,
        input integer minimum_mbps,
        input integer minimum_ddr_efficiency_ppm
    );
        integer total_cycles;
        integer output_span;
        integer response_span;
        integer throughput_mbps;
        integer output_row_efficiency_ppm;
        integer ddr_efficiency_ppm;
        begin
            rst_n = 1'b0;
            profile = selected_profile;
            response_latency = selected_latency;
            clear_metrics();
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            wait (output_count == TOTAL_ROWS);
            repeat (3) @(posedge clk);

            total_cycles = last_output_cycle - start_cycle + 1;
            output_span = last_output_cycle - first_output_cycle + 1;
            throughput_mbps =
                TILE_COUNT * 96 * 100 / total_cycles;
            output_row_efficiency_ppm =
                TOTAL_ROWS * 1_000_000 / output_span;

            if (selected_profile == PROFILE_EXTERNAL_IDEAL) begin
                ddr_efficiency_ppm = 0;
                if (request_count != 0 || response_count != 0)
                    $fatal(1, "%s unexpectedly used DDR", profile_name);
            end else begin
                response_span =
                    last_response_cycle - first_request_cycle + 1;
                ddr_efficiency_ppm =
                    TOTAL_WORDS * 1_000_000 / response_span;
                if (request_count != TOTAL_WORDS ||
                    response_count != TOTAL_WORDS)
                    $fatal(1,
                        "%s direct count requests=%0d responses=%0d",
                        profile_name, request_count, response_count);
            end

            $display(
                "TILELOADER_EFFICIENCY profile=%s tiles=%0d total_cycles=%0d throughput_mbps=%0d ddr_efficiency_ppm=%0d output_row_efficiency_ppm=%0d first_tile_latency=%0d max_tile_gap=%0d descriptor_stall_cycles=%0d request_stall_cycles=%0d response_idle_cycles=%0d output_bubble_cycles=%0d ready_empty_cycles=%0d max_ready_tiles=%0d",
                profile_name, TILE_COUNT, total_cycles, throughput_mbps,
                ddr_efficiency_ppm, output_row_efficiency_ppm,
                first_output_cycle - start_cycle, max_done_gap,
                descriptor_stall_cycles, request_stall_cycles,
                response_idle_cycles, output_bubble_cycles,
                ready_empty_cycles, max_ready_tiles
            );

            if (error || done_count != TILE_COUNT ||
                allocated_tiles != 0 || ready_tiles != 0 ||
                outstanding_words != 0)
                $fatal(1,
                    "%s final state error=%0b done=%0d alloc=%0d ready=%0d outstanding=%0d",
                    profile_name, error, done_count, allocated_tiles,
                    ready_tiles, outstanding_words);
            if (throughput_mbps < minimum_mbps)
                $fatal(1, "%s throughput %0d MB/s below %0d MB/s",
                       profile_name, throughput_mbps, minimum_mbps);
            if (ddr_efficiency_ppm < minimum_ddr_efficiency_ppm)
                $fatal(1,
                    "%s DDR efficiency %0d ppm below %0d ppm",
                    profile_name, ddr_efficiency_ppm,
                    minimum_ddr_efficiency_ppm);
        end
    endtask

    initial begin
        run_profile(
            PROFILE_DIRECT_IDEAL, 2, "direct_ideal",
            370, 950_000
        );
        run_profile(
            PROFILE_DIRECT_CONSTRAINED, 12, "direct_constrained",
            300, 840_000
        );
        run_profile(
            PROFILE_EXTERNAL_IDEAL, 0, "external_ideal",
            350, 0
        );
        $display("INT4 look-ahead TileLoader efficiency benchmark passed");
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk);
        $fatal(1, "TileLoader efficiency benchmark timed out");
    end
endmodule
