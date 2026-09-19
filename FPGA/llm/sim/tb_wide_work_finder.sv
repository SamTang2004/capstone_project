`timescale 1ns/1ps
module tb_wide_work_finder #(parameter bit MERGE_BANKS = 0, parameter bit PIPELINE_MERGE = 0);
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, in_valid = 0, out_ready = 1;
    logic [63:0] in_mask = 0;
    logic [3:0] in_shift = 0;
    logic [511:0] in_values = 0, in_payload = 0;
    logic [15:0] in_meta = 0;
    wire in_ready, out_valid, out_frame_last;
    wire [3:0] out_count;
    wire [63:0] out_values, out_payload;
    wire [47:0] out_source_indices;
    wire [15:0] out_meta;
    llm_wide_work_finder #(.SCAN_LANES(64), .META_WIDTH(16), .MERGE_BANKS(MERGE_BANKS), .PIPELINE_MERGE(PIPELINE_MERGE)) dut (.*);
    wire [7:0] bank_ready_check;
    wire bank_accept_check;
    if (PIPELINE_MERGE) assign bank_accept_check = dut.pipelined.finder.accept;
    else assign bank_accept_check = in_valid && in_ready;
    if (PIPELINE_MERGE) assign bank_ready_check = dut.pipelined.finder.bank_ready;
    else assign bank_ready_check = dut.legacy.bank_ready;
    logic [21:0] entries [0:199999];
    integer ends [0:9999];
    logic [15:0] metas [0:9999];
    integer eh=0, et=0, fh=0, ft=0, cycles=0, frames=0, chunks=0, stalls=0, replacements=0;
    bit accepted, held=0, random_stalls=1;
    logic [196:0] held_packet;
    wire [196:0] packet = {out_count, out_values, out_payload, out_source_indices, out_meta, out_frame_last};
    logic [31:0] rng=32'h8f031de7;
    task automatic random_next;
        rng=rng^(rng<<13); rng=rng^(rng>>17); rng=rng^(rng<<5);
    endtask
    task automatic tick;
        integer v, mag;
        logic [21:0] actual;
        @(posedge clk);
        accepted=rst_n && in_valid && in_ready;
        if (!rst_n) begin
            eh=0; et=0; fh=0; ft=0; held=0;
            if (in_ready || out_valid) $fatal(1,"reset handshake");
        end else begin
            if(bank_accept_check && bank_ready_check !== 8'hff) $fatal(1,"frame accepted by only some banks");
            if (held && (!out_valid || packet !== held_packet)) $fatal(1,"stalled chunk changed");
            if (accepted) begin
                // An accepted frame must be accepted by every local bank.
                for (integer i=0;i<64;i++) begin
                    v=in_values[i*8+:8]; if(v>=128) v-=256;
                    mag=v<0 ? -v : v;
                    if(in_mask[i] && mag >= (1<<in_shift)) begin
                        entries[et]={6'(i),in_payload[i*8+:8],in_values[i*8+:8]}; et++;
                    end
                end
                ends[ft]=et; metas[ft]=in_meta; ft++; frames++;
            end
            if(out_valid) begin
                if(fh==ft || out_meta !== metas[fh] || out_count>8 || eh+out_count>ends[fh])
                    $fatal(1,"bad frame/metadata/count");
                if(out_frame_last !== (eh+out_count==ends[fh])) $fatal(1,"bad frame end");
                if(!out_count && !out_frame_last) $fatal(1,"empty intermediate chunk");
                if(MERGE_BANKS && !out_frame_last && out_count != 8) $fatal(1,"merge left an intermediate chunk partial");
                for(integer i=0;i<8;i++) begin
                    actual={out_source_indices[i*6+:6],out_payload[i*8+:8],out_values[i*8+:8]};
                    if(i<out_count) begin
                        if(actual !== entries[eh+i]) $fatal(1,"lost/reordered/mispaired survivor");
                    end else if(actual !== 0) $fatal(1,"dirty unused slot");
                end
                if(out_ready) begin
                    chunks++; eh+=out_count;
                    if(out_frame_last) begin fh++; if(accepted) replacements++; end
                end else stalls++;
            end
            held=out_valid && !out_ready; held_packet=packet;
        end
        cycles++; #1; @(negedge clk);
    endtask
    task automatic send_frame;
        integer waited;
        in_valid=1; waited=0;
        do begin
            if(random_stalls) begin random_next(); out_ready=rng[0]; end
            tick(); waited++;
            if(waited>2000) $fatal(1,"finder input deadlock");
        end while(!accepted);
        in_valid=0; in_meta++;
    endtask
    initial begin
        tick(); rst_n=1;
        // Every byte value at every position, with every representable threshold.
        for(integer shift=0;shift<16;shift++) begin
            for(integer sample=0;sample<256;sample++) begin
                in_shift=shift; in_mask='1;
                for(integer i=0;i<64;i++) begin
                    in_values[i*8+:8]=sample+i*17;
                    in_payload[i*8+:8]=sample^i;
                end
                send_frame();
            end
        end
        // Each possible isolated source index, including last bank/last lane.
        in_shift=7;
        for(integer i=0;i<64;i++) begin
            in_mask=64'b1<<i; in_values={64{8'h80}}; send_frame();
        end
        // Random masks and backpressure with stable source packets.
        for(integer trial=0;trial<500;trial++) begin
            random_next(); in_mask[31:0]=rng; random_next(); in_mask[63:32]=rng;
            in_shift=rng[3:0];
            for(integer i=0;i<64;i++) begin
                random_next(); in_values[i*8+:8]=rng[7:0]; in_payload[i*8+:8]=rng[15:8];
            end
            send_frame();
        end
        if (MERGE_BANKS) begin
            // Every tail length, spread across banks, including 8->9 and 63->64.
            in_shift=0; in_values={64{8'h03}};
            for(integer count=0;count<=64;count++) begin
                in_mask=0;
                for(integer i=0;i<count;i++) in_mask[(i*17)%64]=1;
                send_frame();
            end
        end
        random_stalls=0; out_ready=1;
        while(fh!=ft) tick();
        // Reset a dense frame while its first chunk is stalled.
        in_mask='1; in_shift=0; in_values={64{8'h03}};
        out_ready=0; send_frame(); repeat(8) tick(); rst_n=0; tick(); rst_n=1;
        out_ready=1; in_mask=0; send_frame(); while(fh!=ft) tick();
        if (MERGE_BANKS) begin
            // Reset after the offset has advanced, with the second chunk held.
            in_mask='1; send_frame(); while(!out_valid) tick(); tick(); out_ready=0; repeat(4) tick();
            rst_n=0; tick(); rst_n=1; out_ready=1;
            in_mask=64'h0101010101010101; send_frame(); while(fh!=ft) tick();
        end
        if (PIPELINE_MERGE) begin
            // Fill every frame stage behind a blocked serializer, then recover.
            out_ready=0; in_mask='1;
            repeat(6) send_frame(); repeat(12) tick();
            if(in_ready) $fatal(1,"full pipeline failed to backpressure input");
            out_ready=1; while(fh!=ft) tick();
            // Reset with several different frame metadata values in flight.
            out_ready=0; repeat(6) send_frame(); repeat(12) tick();
            rst_n=0; tick(); rst_n=1; out_ready=1;
            in_mask=0; send_frame(); while(fh!=ft) tick();
            // A sparse frame stream should sustain one input frame per cycle.
            in_mask=64'h0101010101010101;
            repeat(100) begin
                if(!in_ready) $fatal(1,"sparse pipeline introduced admission bubble");
                send_frame();
            end
            while(fh!=ft) tick();
        end
        repeat(3) tick();
        if(!stalls || !replacements) $fatal(1,"missing flow-control coverage");
        $display("PASS: wide finder frames=%0d chunks=%0d cycles=%0d stalls=%0d replacements=%0d",frames,chunks,cycles,stalls,replacements);
        $finish;
    end
endmodule
