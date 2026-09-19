`timescale 1ns/1ps
module tb_sparse_dot8;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0;
    logic [2:0] cmd_valid = 0, in_valid = 0, result_ready = 3'b111;
    wire [2:0] cmd_ready, in_ready, result_valid, debug_issue;
    logic [7:0] cmd_job_mask [0:2], in_mask [0:2];
    logic [255:0] cmd_job_tags [0:2];
    logic [15:0] cmd_wave_tag [0:2];
    logic [2:0] cmd_w4 = 0, in_job_last = 0, in_wave_last = 0;
    logic [63:0] in_a [0:2], in_b [0:2];
    logic [3:0] in_shift [0:2];
    logic [2:0] in_job_slot [0:2];
    wire [7:0] result_mask [0:2], debug_issue_mask [0:2], debug_useful_mask [0:2];
    wire [255:0] result_tags [0:2], result_values [0:2];
    wire [15:0] result_wave_tag [0:2];
    for (genvar g = 0; g < 3; g++) begin : modes
        llm_sparse_dot8 #(.MODE(g)) dut (
            .clk, .rst_n, .cmd_valid(cmd_valid[g]), .cmd_ready(cmd_ready[g]),
            .cmd_job_mask(cmd_job_mask[g]), .cmd_job_tags(cmd_job_tags[g]),
            .cmd_wave_tag(cmd_wave_tag[g]), .cmd_w4(cmd_w4[g]),
            .in_valid(in_valid[g]), .in_ready(in_ready[g]), .in_mask(in_mask[g]),
            .in_a(in_a[g]), .in_b(in_b[g]), .in_shift(in_shift[g]), .in_job_slot(in_job_slot[g]),
            .in_job_last(in_job_last[g]), .in_wave_last(in_wave_last[g]),
            .result_valid(result_valid[g]), .result_ready(result_ready[g]),
            .result_mask(result_mask[g]), .result_tags(result_tags[g]),
            .result_values(result_values[g]), .result_wave_tag(result_wave_tag[g]),
            .debug_issue(debug_issue[g]), .debug_issue_mask(debug_issue_mask[g]),
            .debug_useful_mask(debug_useful_mask[g])
        );
    end
    integer a [0:7][0:139999], b [0:7][0:139999];
    integer perm_a [0:139999], perm_b [0:139999], lengths [0:7];
    logic [255:0] golden, expected_tags;
    logic [7:0] expected_mask;
    integer expected_wave, wave_id = 0, selected = 0, cycles = 0, checked = 0;
    integer start_cycle, first_source, first_result, source_packets, source_slots;
    integer issue_beats, issue_slots, useful_products, empty_beats, stall_cycles;
    integer expected_useful, accepted_cmd, accepted_input, accepted_result;
    bit pending = 0, source_end_seen = 0;
    logic [31:0] rng = 32'h31415926;
    integer csv;
    task automatic random_next;
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 17); rng = rng ^ (rng << 5);
    endtask
    task automatic tick;
        @(posedge clk);
        accepted_cmd = rst_n && cmd_valid[selected] && cmd_ready[selected];
        accepted_input = rst_n && in_valid[selected] && in_ready[selected];
        accepted_result = rst_n && result_valid[selected] && result_ready[selected];
        if (!rst_n) begin
            pending = 0; source_end_seen = 0;
            if (cmd_ready || in_ready || result_valid) $fatal(1, "reset handshake");
        end else begin
            for (integer m = 0; m < 3; m++)
                if (m != selected && (result_valid[m] || debug_issue[m])) $fatal(1, "inactive mode produced work");
            if (accepted_cmd) begin
                if (pending) $fatal(1, "overlapping wave");
                pending = 1; source_end_seen = 0; start_cycle = cycles;
            end
            if (accepted_input) begin
                if (in_wave_last[selected]) source_end_seen = 1;
                if (first_source < 0) first_source = cycles;
                source_packets++;
                for (integer i = 0; i < 8; i++) if (in_mask[selected][i]) source_slots++;
            end
            if (debug_issue[selected]) begin
                issue_beats++;
                if (debug_issue_mask[selected] == 0) empty_beats++;
                for (integer i = 0; i < 8; i++) begin
                    if (debug_issue_mask[selected][i]) issue_slots++;
                    if (debug_useful_mask[selected][i]) useful_products++;
                end
            end
            if (result_valid[selected]) begin
                if (!pending || !source_end_seen || result_mask[selected] !== expected_mask ||
                    result_tags[selected] !== expected_tags || result_wave_tag[selected] !== expected_wave[15:0] ||
                    result_values[selected] !== golden) begin
                    $display("mode=%0d actual=%h expected=%h", selected, result_values[selected], golden);
                    $fatal(1, "incorrect/early wave result cycle=%0d", cycles);
                end
                if (first_result < 0) first_result = cycles;
                if (!result_ready[selected]) stall_cycles++;
                if (accepted_result) begin pending = 0; checked++; end
            end
        end
        cycles++;
        #1; @(negedge clk);
    endtask

    task automatic prepare(input integer k, input integer density_a, input integer density_b,
                           input bit w4, input integer shift, input integer special);
        integer n, temp, index, count_a, count_b, va, vb, magnitude;
        longint signed sum;
        expected_mask = special == 2 ? 8'h01 : 8'hff;
        expected_useful = 0; golden = 0;
        for (integer j = 0; j < 8; j++) begin
            n = k;
            if (special == 1) begin
                case (j)
                    0: n=3; 1: n=1; 2: n=2; 3: n=2;
                    4: n=0; 5: n=7; 6: n=9; 7: n=16;
                endcase
            end
            if (special == 2 && j != 0) n = 0;
            lengths[j] = n;
            for (integer i = 0; i < n; i++) begin perm_a[i]=i; perm_b[i]=i; end
            if (special != 2) begin
                for (integer i = n-1; i > 0; i--) begin
                    random_next(); index = rng % (i+1);
                    temp=perm_a[i]; perm_a[i]=perm_a[index]; perm_a[index]=temp;
                    random_next(); index = rng % (i+1);
                    temp=perm_b[i]; perm_b[i]=perm_b[index]; perm_b[index]=temp;
                end
            end
            count_a = n*density_a/1000; count_b = n*density_b/1000;
            sum = 0;
            for (integer i = 0; i < n; i++) begin
                random_next(); va = 1 + rng % 127; if (rng[12]) va = -va;
                random_next(); vb = w4 ? (1 + rng % 7) : (1 + rng % 127); if (rng[12]) vb = -vb;
                if (perm_a[i] >= count_a) va = 0;
                if (perm_b[i] >= count_b) vb = 0;
                if (special == 2) begin va = -128; vb = -128; end
                // Include signed W4's minimum value, plus arbitrary ignored upper bits in the driver.
                if (w4 && vb != 0 && i % 11 == 0) vb = -8;
                a[j][i] = va; b[j][i] = vb;
                magnitude = va < 0 ? -va : va;
                if (magnitude >= (1 << shift)) begin
                    sum += va * vb;
                    if (va != 0 && vb != 0) expected_useful++;
                end
            end
            golden[j*32 +: 32] = sum[31:0];
        end
    endtask

    task automatic run_case(input string name, input integer mode, input bit w4,
                            input integer shift, input bit stalls);
        integer waited, n, packets_for_job, k, last_job;
        selected = mode; wave_id++; expected_wave = wave_id;
        source_packets=0; source_slots=0; issue_beats=0; issue_slots=0;
        useful_products=0; empty_beats=0; stall_cycles=0; first_source=-1; first_result=-1;
        expected_tags=0;
        for (integer j=0; j<8; j++) expected_tags[j*32 +: 32] = 32'hf0000000 + wave_id*8+j;
        cmd_job_mask[mode] = expected_mask; cmd_job_tags[mode] = expected_tags;
        cmd_wave_tag[mode] = wave_id; cmd_w4[mode] = w4; cmd_valid[mode] = 1;
        do tick(); while (!accepted_cmd);
        cmd_valid[mode]=0;
        // Verify capture: changing command buses must not change the active wave.
        cmd_job_mask[mode]=0; cmd_job_tags[mode]=0; cmd_wave_tag[mode]=0; cmd_w4[mode]=!w4;
        result_ready[mode] = !stalls;
        last_job=0;
        for (integer j=0; j<8; j++) if (expected_mask[j]) last_job=j;
        for (integer j=0; j<8; j++) begin
            if (expected_mask[j]) begin
                n=lengths[j]; packets_for_job=(n+7)/8;
                if (packets_for_job==0) packets_for_job=1;
                for (integer packet=0; packet<packets_for_job; packet++) begin
                    in_valid[mode]=1; in_mask[mode]=0; in_a[mode]=0; in_b[mode]=0;
                    in_shift[mode]=shift; in_job_slot[mode]=j;
                    in_job_last[mode]=packet==packets_for_job-1;
                    in_wave_last[mode]=j==last_job && packet==packets_for_job-1;
                    for (integer lane=0; lane<8; lane++) begin
                        k=packet*8+lane;
                        if (k<n) begin
                            in_mask[mode][lane]=1; in_a[mode][lane*8 +: 8]=a[j][k];
                            in_b[mode][lane*8 +: 8]=w4 ? (8'ha0 | (b[j][k] & 15)) : b[j][k];
                        end
                    end
                    waited=0;
                    do begin
                        tick(); waited++;
                        if (waited>2000) $fatal(1, "input deadlock");
                    end while (!accepted_input);
                    in_valid[mode]=0;
                    if (stalls && packet%3==0 && !in_wave_last[mode]) repeat (2) tick();
                end
            end
        end
        waited=0;
        while (first_result<0) begin
            tick(); waited++;
            if (waited>2000) $fatal(1, "result deadlock");
        end
        if (stalls) begin repeat (5) tick(); result_ready[mode]=1; tick(); end
        if (useful_products != expected_useful) $fatal(1, "useful-product conservation");
        if (mode>0 && issue_slots != expected_useful) $fatal(1, "zero/non-retained product issued in sparse mode");
        $fdisplay(csv, "%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d", name, mode,
            source_packets, source_slots, useful_products, issue_beats, issue_slots, empty_beats,
            first_result-start_cycle, first_result-first_source, stall_cycles);
        $display("BENCH %s mode=%0d cycles=%0d useful=%0d beats=%0d issued=%0d", name, mode,
            first_result-start_cycle, useful_products, issue_beats, issue_slots);
    endtask

    initial begin
        for (integer m=0; m<3; m++) begin
            cmd_job_mask[m]=0; cmd_job_tags[m]=0; cmd_wave_tag[m]=0;
            in_mask[m]=0; in_a[m]=0; in_b[m]=0; in_shift[m]=0; in_job_slot[m]=0;
        end
        csv=$fopen("performance.csv", "w");
        if (!csv) $fatal(1, "cannot open metrics");
        $fdisplay(csv, "case,mode,source_packets,source_slots,useful_products,issue_beats,issued_slots,empty_beats,command_to_result_cycles,input_to_result_cycles,result_stall_cycles");
        // Allow the vendor glbl startup GSR (100 ns) to finish before traffic.
        repeat (20) tick(); rst_n=1;
        prepare(256,1000,1000,0,0,0);
        for (integer m=0;m<3;m++) run_case("dense_w8",m,0,0,0);
        prepare(256,500,1000,0,0,0);
        for (integer m=0;m<3;m++) run_case("half_w8",m,0,0,0);
        prepare(1000,100,1000,0,0,0);
        for (integer m=0;m<3;m++) run_case("ten_percent_w8",m,0,0,0);
        prepare(1000,1,990,0,0,0);
        for (integer m=0;m<3;m++) run_case("point_one_x_99",m,0,0,0);
        prepare(64,0,1000,0,0,0);
        for (integer m=0;m<3;m++) run_case("zero_jobs",m,0,0,0);
        prepare(16,1000,1000,0,0,1);
        for (integer m=0;m<3;m++) run_case("short_jobs",m,0,0,0);
        prepare(64,1000,750,1,2,0);
        for (integer m=0;m<3;m++) run_case("w4_threshold_stalls",m,1,2,1);
        prepare(16,1000,1000,0,8,0);
        for (integer m=0;m<3;m++) run_case("all_thresholded",m,0,8,0);
        prepare(131080,1000,1000,0,0,2);
        if (golden[31:0] !== 32'h80020000) $fatal(1, "overflow fixture invalid");
        for (integer m=0;m<3;m++) run_case("int32_wrap",m,0,0,0);
        // Cancel a wave immediately after accepting a final operand packet.
        for (integer m=0;m<3;m++) begin
            selected=m; expected_mask=1; expected_tags=0; expected_wave=0;
            cmd_job_mask[m]=1; cmd_job_tags[m]=0; cmd_wave_tag[m]=0; cmd_w4[m]=0; cmd_valid[m]=1;
            do tick(); while (!accepted_cmd); cmd_valid[m]=0;
            in_valid[m]=1; in_mask[m]=255; in_a[m]=64'h0101010101010101;
            in_b[m]=64'h0202020202020202; in_shift[m]=0; in_job_slot[m]=0;
            in_job_last[m]=1; in_wave_last[m]=1;
            do tick(); while (!accepted_input); in_valid[m]=0;
            rst_n=0; tick(); rst_n=1; repeat (5) tick();
        end
        prepare(8,1000,1000,0,0,0);
        for (integer m=0;m<3;m++) run_case("after_reset",m,0,0,0);
        // Reuse the same core immediately: previous dispatch fences may still drain.
        prepare(1,1000,1000,0,0,0);
        for (integer w=0;w<4;w++) run_case($sformatf("back_to_back_%0d",w),2,0,0,0);
        $fclose(csv);
        $display("PASS: sparse dot8 real DSP48E1 checked=%0d cycles=%0d",checked,cycles);
        $finish;
    end
endmodule
