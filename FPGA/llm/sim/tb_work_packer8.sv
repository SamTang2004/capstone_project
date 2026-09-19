`timescale 1ns/1ps
module tb_work_packer8;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, in_valid = 0, out_ready = 1, done_ready = 1;
    logic [7:0] in_mask = 0;
    logic [3:0] in_shift = 0;
    logic [63:0] in_values = 0;
    logic [255:0] in_payload = 0;
    logic [31:0] in_job_tag = 0;
    logic in_job_last = 0, in_wave_last = 0;
    logic [15:0] in_wave_tag = 0;
    wire in_ready, out_valid, done_valid, out_wave_last;
    wire [3:0] out_count, pending_jobs;
    wire [7:0] out_mask;
    wire [63:0] out_values;
    wire [255:0] out_payload, out_job_tags;
    wire [23:0] out_source_lanes;
    wire [15:0] out_wave_tag;
    wire [31:0] done_job_tag;
    wire [4:0] occupancy;
    wire [628:0] output_packet = {out_count, out_mask, out_values, out_payload,
        out_source_lanes, out_job_tags, out_wave_last, out_wave_tag};
    llm_work_packer8 dut (.*);

    // Flat logical stream reference; no model of the DUT's buffer or FIFO layout.
    logic [74:0] expected [0:99999];
    integer job_frontier [0:19999], wave_frontier [0:19999];
    logic [31:0] job_tags [0:19999];
    logic [15:0] wave_tags [0:19999];
    integer eh = 0, et = 0, jh = 0, jt = 0, wh = 0, wt = 0;
    integer cycles = 0, packets = 0, dispatched = 0, completions = 0, waves = 0;
    integer mixed = 0, pressure_partials = 0, empty_flushes = 0;
    integer data_stalls = 0, done_stalls = 0, replacements = 0;
    bit was_stalled = 0, done_was_stalled = 0;
    logic [628:0] held_packet;
    logic [31:0] held_done;
    logic [31:0] rng = 32'h47a58bd1;
    bit random_flow = 0, accepted_now;
    integer serial = 0, next_job = 1, next_wave = 1;

    task automatic random_next;
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 17); rng = rng ^ (rng << 5);
    endtask

    task automatic tick;
        integer v, mag, dispatched_before;
        logic [74:0] actual;
        @(posedge clk);
        accepted_now = rst_n && in_valid && in_ready;
        dispatched_before = eh;
        if (!rst_n) begin
            if (in_ready || out_valid || done_valid) $fatal(1, "handshake active during reset");
            eh = 0; et = 0; jh = 0; jt = 0; wh = 0; wt = 0;
            was_stalled = 0; done_was_stalled = 0;
        end else begin
            if (occupancy > 15 || pending_jobs > 8) $fatal(1, "capacity violation");
            if (was_stalled && (!out_valid || output_packet !== held_packet))
                $fatal(1, "output changed while stalled cycle=%0d", cycles);
            if (done_was_stalled && (!done_valid || done_job_tag !== held_done))
                $fatal(1, "completion changed while stalled");
            if (accepted_now) begin
                packets++;
                for (integer i = 0; i < 8; i++) begin
                    v = in_values[i*8 +: 8];
                    if (v >= 128) v = v - 256;
                    mag = v < 0 ? -v : v;
                    if (in_mask[i] && mag >= (1 << in_shift)) begin
                        expected[et] = {in_job_tag, 3'(i), in_payload[i*32 +: 32], in_values[i*8 +: 8]};
                        et++;
                    end
                end
                if (in_job_last) begin job_tags[jt] = in_job_tag; job_frontier[jt] = et; jt++; end
                if (in_wave_last) begin wave_tags[wt] = in_wave_tag; wave_frontier[wt] = et; wt++; end
            end
            if (out_valid) begin
                if (out_count > 8 || (out_count == 0 && !out_wave_last)) $fatal(1, "invalid output count");
                if (out_mask !== ((1 << out_count) - 1)) $fatal(1, "bad output mask");
                if (eh + out_count > et) $fatal(1, "invented survivors");
                if (wh < wt && eh + out_count > wave_frontier[wh]) $fatal(1, "crossed wave boundary");
                for (integer i = 0; i < 8; i++) begin
                    actual = {out_job_tags[i*32 +: 32], out_source_lanes[i*3 +: 3],
                        out_payload[i*32 +: 32], out_values[i*8 +: 8]};
                    if (i < out_count) begin
                        if (actual !== expected[eh+i]) $fatal(1, "survivor mismatch cycle=%0d lane=%0d", cycles, i);
                    end else if (actual !== 75'd0) $fatal(1, "dirty unused slot");
                end
                if (out_wave_last && (wh == wt || out_wave_tag !== wave_tags[wh] ||
                    eh + out_count != wave_frontier[wh])) $fatal(1, "incorrect flush boundary/tag");
                if (out_ready) begin
                    if (out_count == 8 && out_job_tags[31:0] != out_job_tags[255:224]) mixed++;
                    if (out_count < 8 && !out_wave_last) pressure_partials++;
                    if (out_count == 0) empty_flushes++;
                    eh += out_count; dispatched += out_count;
                    if (out_wave_last) begin wh++; waves++; end
                    if (accepted_now) replacements++;
                end else data_stalls++;
            end
            if (done_valid) begin
                if (jh == jt || done_job_tag !== job_tags[jh] || dispatched_before < job_frontier[jh])
                    $fatal(1, "early/wrong job fence cycle=%0d", cycles);
                if (done_ready) begin jh++; completions++; end else done_stalls++;
            end
            was_stalled = out_valid && !out_ready; held_packet = output_packet;
            done_was_stalled = done_valid && !done_ready; held_done = done_job_tag;
        end
        cycles++;
        #1;
        @(negedge clk);
    endtask

    task automatic send_packet(input integer survivors, input integer job,
                               input bit last_job, input bit last_wave, input integer wave);
        integer waited;
        in_valid = 1; in_job_tag = job; in_job_last = last_job;
        in_wave_last = last_wave; in_wave_tag = wave;
        serial++;
        if (survivors >= 0) begin
            in_shift = 0; in_mask = (1 << survivors) - 1;
            for (integer i = 0; i < 8; i++) begin
                in_values[i*8 +: 8] = i + 1;
                in_payload[i*32 +: 32] = serial*8 + i;
            end
        end else begin
            random_next(); in_shift = rng[3:0]; in_mask = rng[15:8];
            for (integer i = 0; i < 8; i++) begin
                random_next(); in_values[i*8 +: 8] = rng[7:0];
                in_payload[i*32 +: 32] = serial*8 + i;
            end
        end
        waited = 0;
        do begin
            if (random_flow) begin random_next(); out_ready = rng[0]; done_ready = rng[1]; end
            tick(); waited++;
            if (waited > 2000) $fatal(1, "input deadlock");
        end while (!accepted_now);
        in_valid = 0;
    endtask

    task automatic drain;
        integer waited;
        in_valid = 0; out_ready = 1; done_ready = 1; waited = 0;
        do begin
            tick(); waited++;
            if (waited > 2000) $fatal(1, "drain deadlock entries=%0d jobs=%0d waves=%0d", et-eh, jt-jh, wt-wh);
        end while (eh != et || jh != jt || wh != wt || out_valid || done_valid);
        repeat (3) tick();
    endtask

    initial begin
        tick(); rst_n = 1;
        // Four independent jobs densely packed into one beat.
        send_packet(3, 1, 1, 0, 1); send_packet(1, 2, 1, 0, 1);
        send_packet(2, 3, 1, 0, 1); send_packet(2, 4, 1, 1, 1); drain();
        // Empty wave and full output already sent before a zero-entry end marker.
        send_packet(0, 5, 1, 1, 2); drain();
        send_packet(8, 6, 0, 0, 3); repeat (5) tick();
        send_packet(0, 6, 1, 1, 3); drain();
        // A single survivor followed by many empty jobs must not deadlock fences.
        send_packet(1, 10, 1, 0, 4);
        for (integer j = 11; j < 24; j++) send_packet(0, j, 1, 0, 4);
        send_packet(0, 0, 0, 1, 4); drain();
        // Pressure-triggered partial output must remain stable under backpressure.
        out_ready = 0;
        send_packet(1, 30, 1, 0, 5);
        for (integer j = 31; j < 38; j++) send_packet(0, j, 1, 0, 5);
        repeat (12) tick();
        out_ready = 1; send_packet(0, 0, 0, 1, 5); drain();
        // Dense steady state must accept a packet every cycle, without bubbles.
        begin : dense_throughput
            integer started;
            started = cycles;
            for (integer c = 0; c < 32; c++) send_packet(8, 40, c == 31, c == 31, 6);
            if (cycles - started != 32) $fatal(1, "dense input throughput regressed");
            drain();
        end
        // All pairs of packet lengths exercise carry-over, exact fill and tails.
        next_job = 100; next_wave = 100;
        for (integer a = 0; a <= 8; a++) begin
            for (integer b = 0; b <= 8; b++) begin
                send_packet(a, next_job, 1, 0, next_wave); next_job++;
                send_packet(b, next_job, 1, 1, next_wave); next_job++; next_wave++;
            end
        end
        drain();
        // Hold full output and completion independently, then reset occupied state.
        out_ready = 0; done_ready = 0;
        send_packet(0, next_job, 1, 0, next_wave); next_job++;
        send_packet(8, next_job, 1, 0, next_wave); next_job++;
        repeat (10) tick();
        rst_n = 0; tick(); rst_n = 1; out_ready = 1; done_ready = 1;
        send_packet(1, next_job, 1, 1, next_wave); next_job++; next_wave++; drain();
        // Reset also cancels a pending partial wave flush and its job fence.
        out_ready = 0;
        send_packet(3, next_job, 1, 1, next_wave); next_job++; next_wave++;
        repeat (6) tick(); rst_n = 0; tick(); rst_n = 1;
        out_ready = 1; send_packet(0, next_job, 1, 1, next_wave); next_job++; next_wave++; drain();
        // Realistic legal source stalls: input and all metadata held until accepted.
        random_flow = 1;
        for (integer w = 0; w < 200; w++) begin
            integer jobs, chunks;
            random_next(); jobs = 1 + rng % 20;
            for (integer j = 0; j < jobs; j++) begin
                random_next(); chunks = 1 + rng % 5;
                for (integer c = 0; c < chunks; c++) begin
                    send_packet(-1, next_job, c == chunks-1, j == jobs-1 && c == chunks-1, next_wave);
                    random_next();
                    if (rng[2:0] == 0) tick();
                end
                next_job++;
            end
            next_wave++;
        end
        random_flow = 0; drain();
        if (!mixed || !pressure_partials || !empty_flushes || !data_stalls || !done_stalls || !replacements)
            $fatal(1, "missing required coverage");
        $display("PASS: work packer8 cycles=%0d packets=%0d survivors=%0d jobs=%0d waves=%0d", cycles, packets, dispatched, completions, waves);
        $display("Coverage mixed_full=%0d pressure_partial=%0d empty_flush=%0d data_stall=%0d done_stall=%0d concurrent_io=%0d", mixed, pressure_partials, empty_flushes, data_stalls, done_stalls, replacements);
        $finish;
    end
endmodule
