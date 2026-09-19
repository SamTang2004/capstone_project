`timescale 1ns/1ps
module tb_work_finder8;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, in_valid = 0, out_ready = 0;
    logic [7:0] in_mask = 0;
    logic [3:0] in_shift = 0;
    logic [63:0] in_values = 0;
    logic [255:0] in_payload = 0;
    wire [1:0] in_ready, out_valid;
    wire [3:0] out_count [0:1];
    wire [7:0] out_mask [0:1], out_keep_mask [0:1];
    wire [63:0] out_values [0:1];
    wire [255:0] out_payload [0:1];
    wire [23:0] out_source_lanes [0:1];
    for (genvar g = 0; g < 2; g++) begin : dut_modes
        llm_work_finder8 #(.SIGNED_MAGNITUDE(g)) dut (
            .clk, .rst_n, .in_valid, .in_ready(in_ready[g]),
            .in_mask, .in_shift, .in_values, .in_payload,
            .out_valid(out_valid[g]), .out_ready,
            .out_count(out_count[g]), .out_mask(out_mask[g]),
            .out_keep_mask(out_keep_mask[g]), .out_values(out_values[g]),
            .out_payload(out_payload[g]), .out_source_lanes(out_source_lanes[g])
        );
    end
    bit expected_valid = 0;
    integer counts [0:1];
    logic [7:0] masks [0:1], keeps [0:1];
    logic [63:0] values [0:1];
    logic [255:0] payloads [0:1];
    logic [23:0] sources [0:1];
    integer accepted = 0, cycles = 0;
    integer m, i, v, magnitude, n, x, mask;
    logic [31:0] rng = 32'h91e10da5;

    task automatic random_next;
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 17);
        rng = rng ^ (rng << 5);
    endtask

    // Independent integer-threshold reference, with elastic-register scoreboard.
    task automatic tick;
        @(posedge clk);
        if (in_ready !== {2{rst_n && (!expected_valid || out_ready)}})
            $fatal(1, "ready mismatch at cycle %0d", cycles);
        if (!rst_n) expected_valid = 0;
        else if (!expected_valid || out_ready) begin
            expected_valid = in_valid;
            if (in_valid) begin
                accepted++;
                for (m = 0; m < 2; m++) begin
                    counts[m] = 0; masks[m] = 0; keeps[m] = 0;
                    values[m] = 0; payloads[m] = 0; sources[m] = 0;
                    for (i = 0; i < 8; i++) begin
                        v = in_values[i*8 +: 8];
                        if (m == 1 && v >= 128) v = v - 256;
                        magnitude = v < 0 ? -v : v;
                        if (in_mask[i] && magnitude >= (1 << in_shift)) begin
                            n = counts[m];
                            masks[m][n] = 1; keeps[m][i] = 1;
                            values[m][n*8 +: 8] = in_values[i*8 +: 8];
                            payloads[m][n*32 +: 32] = in_payload[i*32 +: 32];
                            sources[m][n*3 +: 3] = i;
                            counts[m]++;
                        end
                    end
                end
            end
        end
        #1;
        if (out_valid !== {2{expected_valid && rst_n}})
            $fatal(1, "valid mismatch");
        if (expected_valid) begin
            for (m = 0; m < 2; m++) begin
                if (out_count[m] !== counts[m] || out_mask[m] !== masks[m] ||
                    out_keep_mask[m] !== keeps[m] || out_values[m] !== values[m] ||
                    out_payload[m] !== payloads[m] || out_source_lanes[m] !== sources[m])
                    $fatal(1, "packet mismatch cycle=%0d mode=%0d", cycles, m);
            end
        end
        cycles++;
        @(negedge clk);
    endtask

    initial begin
        tick(); rst_n = 1; in_valid = 1; out_ready = 1; in_mask = 255;
        // Every possible 8-bit value at every lane, every supported shift code.
        for (x = 0; x < 16; x++) begin
            in_shift = x;
            for (integer sample = 0; sample < 256; sample++) begin
                for (integer lane = 0; lane < 8; lane++) begin
                    in_values[lane*8 +: 8] = (sample + lane*31) % 256;
                    in_payload[lane*32 +: 32] = sample*8 + lane;
                end
                tick();
            end
        end
        // All keep-mask shapes with nonzero data, including empty/full packets.
        in_shift = 0; in_values = 64'h0102030405060708;
        for (mask = 0; mask < 256; mask++) begin in_mask = mask; tick(); end
        // Random input bubbles, downstream stalls, and reset of occupied output.
        for (integer trial = 0; trial < 3000; trial++) begin
            random_next(); in_valid = rng[0]; out_ready = rng[1];
            in_mask = rng[15:8]; in_shift = rng[19:16];
            rst_n = (trial % 127 != 0);
            for (integer lane = 0; lane < 8; lane++) begin
                random_next(); in_values[lane*8 +: 8] = rng[7:0];
                in_payload[lane*32 +: 32] = rng;
            end
            tick();
        end
        rst_n = 1; in_valid = 0; out_ready = 1; tick(); tick();
        $display("PASS: work finder8 cycles=%0d accepted=%0d, raw and magnitude modes", cycles, accepted);
        $finish;
    end
endmodule
