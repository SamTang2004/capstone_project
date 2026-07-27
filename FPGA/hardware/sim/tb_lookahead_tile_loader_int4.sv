`timescale 1ns/1ps
module tb_lookahead_tile_loader_int4;
    localparam int DESCRIPTOR_COUNT = 6;
    localparam int REQUEST_COUNT = 218;
    localparam int OUTPUT_COUNT = DESCRIPTOR_COUNT * 8;
    localparam string DESCRIPTOR_PATH =
        "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/lookahead_tileloader_descriptors.hex";
    localparam string REQUEST_PATH =
        "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/lookahead_tileloader_requests.hex";
    localparam string OUTPUT_PATH =
        "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/lookahead_tileloader_outputs.hex";

    logic clk = 1'b0;
    logic rst_n = 1'b0;

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

    logic read_req_valid;
    logic read_req_ready;
    logic [31:0] read_req_addr;
    logic [4:0] read_req_len;
    logic read_word_valid;
    logic read_word_ready;
    logic [31:0] read_word_data;
    logic [3:0] read_word_keep = 4'hf;
    logic read_word_last = 1'b1;

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

    logic [255:0] descriptors [0:DESCRIPTOR_COUNT-1];
    logic [31:0] expected_requests [0:REQUEST_COUNT-1];
    logic [175:0] expected_outputs [0:OUTPUT_COUNT-1];
    logic [31:0] response_addresses [0:255];

    integer cycle_count = 0;
    integer descriptor_index = 0;
    integer request_index = 0;
    integer response_write = 0;
    integer response_read = 0;
    integer output_index = 0;
    integer done_count = 0;
    integer source_stall_cycles = 0;
    integer request_stall_cycles = 0;
    integer output_stall_cycles = 0;
    integer max_allocated = 0;
    logic full_observed = 1'b0;
    logic drain_enabled = 1'b0;
    logic stalled_output_valid = 1'b0;
    logic [175:0] stalled_output;

    always #5 clk = ~clk;

    function automatic logic [7:0] memory_byte(
        input logic [31:0] address
    );
        memory_byte =
            ((address * 29) ^ (address >> 3) ^ 32'h0000_00a7) & 8'hff;
    endfunction

    function automatic logic [31:0] memory_word(
        input logic [31:0] address
    );
        for (int lane = 0; lane < 4; lane++)
            memory_word[lane*8 +: 8] = memory_byte(address + lane);
    endfunction

    wire [255:0] current_descriptor =
        descriptor_index < DESCRIPTOR_COUNT ?
        descriptors[descriptor_index] : '0;

    assign descriptor_valid = descriptor_index < DESCRIPTOR_COUNT;
    assign descriptor_id = current_descriptor[255:240];
    assign descriptor_a_base = current_descriptor[239:208];
    assign descriptor_b0_base = current_descriptor[207:176];
    assign descriptor_b1_base = current_descriptor[175:144];
    assign descriptor_a_stride = current_descriptor[143:128];
    assign descriptor_b0_stride = current_descriptor[127:112];
    assign descriptor_b1_stride = current_descriptor[111:96];
    assign descriptor_a_row_mask = current_descriptor[95:88];
    assign descriptor_b_row_mask = current_descriptor[87:80];
    assign descriptor_a_lane_mask = current_descriptor[79:72];
    assign descriptor_b0_lane_mask = current_descriptor[71:64];
    assign descriptor_b1_lane_mask = current_descriptor[63:56];

    assign read_req_ready = rst_n && cycle_count % 7 != 2 &&
                            cycle_count % 7 != 5;
    assign read_word_valid = response_read < response_write &&
                             cycle_count % 5 != 1;
    assign read_word_data = memory_word(response_addresses[response_read]);
    assign output_ready = drain_enabled &&
                          cycle_count % 9 != 3 &&
                          cycle_count % 9 != 4;

    lookahead_tile_loader_int4 #(
        .TILE_BANKS(4)
    ) dut (
        .clk,
        .rst_n,
        .descriptor_valid,
        .descriptor_ready,
        .descriptor_id,
        .descriptor_a_base,
        .descriptor_b0_base,
        .descriptor_b1_base,
        .descriptor_a_stride,
        .descriptor_b0_stride,
        .descriptor_b1_stride,
        .descriptor_a_row_mask,
        .descriptor_b_row_mask,
        .descriptor_a_lane_mask,
        .descriptor_b0_lane_mask,
        .descriptor_b1_lane_mask,
        .descriptor_external_fill(1'b0),
        .external_vector_valid(1'b0),
        .external_vector_ready(),
        .external_vector_array(1'b0),
        .external_vector_matrix(1'b0),
        .external_vector_row(3'd0),
        .external_vector_data(32'd0),
        .external_commit_valid(1'b0),
        .external_commit_ready(),
        .read_req_valid,
        .read_req_ready,
        .read_req_addr,
        .read_req_len,
        .read_word_valid,
        .read_word_ready,
        .read_word_data,
        .read_word_keep,
        .read_word_last,
        .output_valid,
        .output_ready,
        .output_descriptor_id,
        .output_row,
        .output_a_mask,
        .output_a_data,
        .output_b0_mask,
        .output_b0_data,
        .output_b1_mask,
        .output_b1_data,
        .group_done,
        .allocated_tiles,
        .ready_tiles,
        .outstanding_words,
        .cache_full,
        .error
    );

    always_ff @(posedge clk) begin
        logic [175:0] actual_output;

        cycle_count <= cycle_count + 1;
        if (!rst_n) begin
            descriptor_index <= 0;
            request_index <= 0;
            response_write <= 0;
            response_read <= 0;
            output_index <= 0;
            done_count <= 0;
            source_stall_cycles <= 0;
            request_stall_cycles <= 0;
            output_stall_cycles <= 0;
            max_allocated <= 0;
            full_observed <= 1'b0;
            drain_enabled <= 1'b0;
            stalled_output_valid <= 1'b0;
        end else begin
            if (descriptor_valid && descriptor_ready)
                descriptor_index <= descriptor_index + 1;
            if (descriptor_valid && !descriptor_ready)
                source_stall_cycles <= source_stall_cycles + 1;

            if (read_req_valid && read_req_ready) begin
                if (request_index >= REQUEST_COUNT)
                    $fatal(1, "TileLoader issued too many requests");
                if (read_req_addr !== expected_requests[request_index])
                    $fatal(1,
                        "request %0d mismatch expected=%08x got=%08x",
                        request_index, expected_requests[request_index],
                        read_req_addr);
                if (read_req_len != 5'd4)
                    $fatal(1, "request %0d was not one aligned word",
                           request_index);
                response_addresses[response_write] <= read_req_addr;
                response_write <= response_write + 1;
                request_index <= request_index + 1;
            end
            if (read_req_valid && !read_req_ready)
                request_stall_cycles <= request_stall_cycles + 1;
            if (read_word_valid && read_word_ready)
                response_read <= response_read + 1;

            if (allocated_tiles > max_allocated)
                max_allocated <= allocated_tiles;
            if (cache_full)
                full_observed <= 1'b1;
            if (request_index >= 96)
                drain_enabled <= 1'b1;

            actual_output = {
                output_descriptor_id,
                output_row,
                output_a_mask,
                output_b0_mask,
                output_b1_mask,
                37'b0,
                output_b1_data,
                output_b0_data,
                output_a_data
            };

            if (output_valid && !output_ready) begin
                output_stall_cycles <= output_stall_cycles + 1;
                if (!stalled_output_valid) begin
                    stalled_output_valid <= 1'b1;
                    stalled_output <= actual_output;
                end else if (actual_output !== stalled_output) begin
                    $fatal(1, "output payload changed under backpressure");
                end
            end else begin
                stalled_output_valid <= 1'b0;
            end

            if (output_valid && output_ready) begin
                if (output_index >= OUTPUT_COUNT)
                    $fatal(1, "TileLoader produced too many rows");
                if (actual_output !== expected_outputs[output_index])
                    $fatal(1,
                        "output %0d mismatch expected=%044x got=%044x",
                        output_index, expected_outputs[output_index],
                        actual_output);
                output_index <= output_index + 1;
            end

            if (group_done)
                done_count <= done_count + 1;
        end
    end

    initial begin
        $readmemh(DESCRIPTOR_PATH, descriptors);
        $readmemh(REQUEST_PATH, expected_requests);
        $readmemh(OUTPUT_PATH, expected_outputs);

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;

        wait (output_index == OUTPUT_COUNT);
        repeat (3) @(posedge clk);

        if (error)
            $fatal(1, "TileLoader reported a protocol error");
        if (descriptor_index != DESCRIPTOR_COUNT ||
            request_index != REQUEST_COUNT ||
            response_read != REQUEST_COUNT ||
            done_count != DESCRIPTOR_COUNT)
            $fatal(1,
                "count mismatch descriptors=%0d requests=%0d responses=%0d groups=%0d",
                descriptor_index, request_index, response_read, done_count);
        if (allocated_tiles != 0 || ready_tiles != 0 ||
            outstanding_words != 0)
            $fatal(1,
                "loader did not drain allocated=%0d ready=%0d outstanding=%0d",
                allocated_tiles, ready_tiles, outstanding_words);
        if (!full_observed || max_allocated != 4 ||
            source_stall_cycles == 0 || request_stall_cycles == 0 ||
            output_stall_cycles == 0)
            $fatal(1,
                "stress coverage missing full=%0b max=%0d source=%0d request=%0d output=%0d",
                full_observed, max_allocated, source_stall_cycles,
                request_stall_cycles, output_stall_cycles);

        $display(
            "INT4 look-ahead TileLoader golden trace passed: 6 descriptors, 4 credited BRAM banks, 218 aligned gather reads, 48 stable SIMD rows"
        );
        $finish;
    end

    initial begin
        repeat (1500) @(posedge clk);
        $fatal(1, "look-ahead TileLoader timed out");
    end
endmodule
