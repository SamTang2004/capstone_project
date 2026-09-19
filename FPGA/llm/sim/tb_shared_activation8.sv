`timescale 1ns/1ps
module tb_shared_activation8 #(parameter bit BITMAP=0, parameter bit STREAMING=0);
    logic clk=0; always #5 clk=~clk;
    logic rst_n=0, cmd_valid=0, in_valid=0, in_last=0;
    wire cmd_ready, in_ready;
    logic [3:0] cmd_rows,cmd_passes,cmd_shift;
    logic [511:0] in_values=0;
    logic [63:0] in_mask=0;
    wire req_valid,req_last,result_valid,rsp_ready;
    logic req_ready=0,rsp_valid=0,rsp_last=0,result_ready=1;
    wire [11:0] req_index;
    wire [7:0] req_activation,req_mask;
    logic [7:0] rsp_activation=0,rsp_mask=0;
    logic [63:0] rsp_weights=0;
    wire [255:0] result_values;
    wire debug_scanning,debug_scan_accept,debug_descriptor_write,debug_issue;
    wire [12:0] debug_descriptor_count;
    llm_shared_activation8 #(.BITMAP(BITMAP),.STREAMING(STREAMING)) dut (.*);
    integer x[0:4095],w[0:7][0:4095],permutation[0:4095];
    integer expected_indices[0:4095];
    logic [255:0] golden;
    integer length,rows,shift,kept,expected_useful;
    integer cycle=0,start_cycle,first_result,scan_cycles,scan_frames,scan_stalls;
    integer writes,requests,weight_bytes,issues,useful,checked=0,total_dots=0;
    integer request_pos, result_stalls;
    integer q_head=0,q_tail=0,q_count=0;
    integer due[0:7];
    logic [7:0] qa[0:7],qm[0:7];
    logic [63:0] qw[0:7];
    logic qlast[0:7];
    bit stalls=0, pending=0, aborted=0;
    integer accepted_cmd,accepted_input,accepted_result;
    logic held_req=0;
    logic [29:0] previous_req;
    logic held_result=0;
    logic [255:0] previous_result;
    logic [31:0] rng=32'h617A93B1;
    integer csv;
    task automatic random_next;
        rng=rng^(rng<<13); rng=rng^(rng>>17); rng=rng^(rng<<5);
    endtask
    task automatic tick;
        req_ready = q_count<8 && (!stalls || cycle%7!=0);
        rsp_valid = q_count>0 && due[q_head]<=cycle;
        rsp_activation=qa[q_head]; rsp_mask=qm[q_head];
        rsp_weights=qw[q_head]; rsp_last=qlast[q_head];
        result_ready = !stalls || result_stalls>=7;
        @(posedge clk);
        accepted_cmd=rst_n && cmd_valid && cmd_ready;
        accepted_input=rst_n && in_valid && in_ready;
        accepted_result=rst_n && result_valid && result_ready;
        if (rst_n) begin
            if (held_req && {req_valid,req_index,req_activation,req_mask,req_last} !== previous_req)
                $fatal(1,"request changed under backpressure");
            if (held_result && (!result_valid || result_values!==previous_result))
                $fatal(1,"result changed under backpressure");
            if (accepted_cmd) begin pending=1; start_cycle=cycle; end
            if (debug_scanning) scan_cycles++;
            if (debug_scan_accept) scan_frames++;
            if (in_valid && !in_ready) scan_stalls++;
            if (debug_descriptor_write) writes++;
            if (rsp_valid && rsp_ready) begin
                issues++;
                for(integer r=0;r<8;r++) if(rsp_mask[r] && rsp_activation!=0 && rsp_weights[r*8+:8]!=0) useful++;
                q_head=(q_head+1)%8; q_count--;
            end
            if (req_valid && req_ready) begin
                if (req_mask==0) begin
                    if(!req_last || request_pos!=kept) $fatal(1,"empty marker");
                end else begin
                    if(request_pos>=kept || req_index!=expected_indices[request_pos] ||
                       $signed(req_activation)!=x[req_index] || req_mask!=8'((1<<rows)-1) ||
                       (!STREAMING && req_last!=(request_pos==kept-1)) ||
                       (req_last && request_pos!=kept-1)) $fatal(1,"descriptor/index/last mismatch");
                    request_pos++;
                end
                requests++;
                qa[q_tail]=req_activation; qm[q_tail]=req_mask; qlast[q_tail]=req_last;
                qw[q_tail]=0;
                for(integer r=0;r<8;r++) if(req_mask[r]) begin
                    qw[q_tail][r*8+:8]=8'(w[r][req_index]); weight_bytes++;
                end
                // Output-interleaved layout: each output row owns one bank.
                due[q_tail]=cycle+2; q_tail=(q_tail+1)%8; q_count++;
                if(q_count>8) $fatal(1,"memory response queue overflow");
            end
            if(result_valid) begin
                if(!pending || result_values!==golden || q_count!=0) begin
                    $display("actual=%h expected=%h",result_values,golden);
                    $fatal(1,"incorrect/early result");
                end
                if(first_result<0) first_result=cycle;
                if(!result_ready) result_stalls++;
                if(accepted_result) begin pending=0; checked++; total_dots+=rows; end
            end
        end
        held_req = rst_n && req_valid && !req_ready;
        previous_req={req_valid,req_index,req_activation,req_mask,req_last};
        held_result=rst_n && result_valid && !result_ready;
        previous_result=result_values;
        cycle++;
        if(cycle>3000000) $fatal(1,"timeout");
        #1; @(negedge clk);
    endtask
    task automatic reset;
        rst_n=0; cmd_valid=0; in_valid=0; q_count=0; q_head=0; q_tail=0;
        pending=0; held_req=0; held_result=0;
        repeat(4) tick(); rst_n=1;
    endtask
    task automatic prepare(input integer n, row_count, density, xshift, special);
        integer j,tmp,v;
        longint signed sum;
        length=n; rows=row_count; shift=xshift; kept=0; golden=0; expected_useful=0;
        rng=32'h617A93B1;
        for(integer k=0;k<n;k++) permutation[k]=k;
        for(integer k=n-1;k>0;k--) begin
            random_next(); j=rng%(k+1); tmp=permutation[k]; permutation[k]=permutation[j]; permutation[j]=tmp;
        end
        for(integer k=0;k<n;k++) begin
            random_next(); v=1+rng%127; if(rng[9]) v=-v;
            x[k]=permutation[k]<n*density/1000 ? v : 0;
            if(special==1) x[k]=(k%5==0) ? -128 : (k%5==1) ? -1 : (k%5==2) ? 0 : (k%5==3) ? 1 : 127;
            if((x[k]<0 ? -x[k] : x[k]) >= (1<<shift)) begin expected_indices[kept]=k; kept++; end
            for(integer r=0;r<8;r++) begin
                random_next(); v=1+rng%127; if(rng[11]) v=-v;
                if(special==1) v=(k+r)%16-8; // Includes W4 minimum and zero.
                w[r][k]=v;
            end
        end
        for(integer r=0;r<rows;r++) begin
            sum=0;
            for(integer k=0;k<n;k++) if((x[k]<0 ? -x[k] : x[k]) >= (1<<shift)) begin
                sum+=x[k]*w[r][k]; if(x[k]!=0 && w[r][k]!=0) expected_useful++;
            end
            golden[r*32+:32]=sum[31:0];
        end
    endtask
    task automatic run_case(input string name,input integer passes,bw,input bit stalled,input integer abort_at);
        integer frames,n;
        scan_cycles=0; scan_frames=0; scan_stalls=0; writes=0; requests=0;
        weight_bytes=0; issues=0; useful=0; request_pos=0; result_stalls=0;
        first_result=-1; stalls=stalled; aborted=0;
        cmd_rows=4'(rows); cmd_passes=4'(passes); cmd_shift=4'(shift); cmd_valid=1;
        do tick(); while(!accepted_cmd);
        cmd_valid=0;
        frames=length==0 ? 1 : (length+63)/64;
        for(integer p=0;p<passes;p++) begin
            for(integer f=0;f<frames;f++) begin
                n=length-f*64; if(n>64)n=64;
                repeat((n+bw-1)/bw > 0 ? (n+bw-1)/bw-1 : 0) tick();
                in_values=0; in_mask=0; in_last=f==frames-1;
                for(integer i=0;i<n;i++) begin in_values[i*8+:8]=8'(x[f*64+i]); in_mask[i]=1; end
                in_valid=1; do tick(); while(!accepted_input); in_valid=0;
                if(abort_at==1) begin aborted=1; reset(); return; end
                if(stalls && f%3==0) repeat(2) tick();
            end
        end
        if(abort_at==2) begin
            while(requests<4) tick();
            aborted=1; reset(); return;
        end
        while(pending) tick();
        if(writes!=passes*kept || request_pos!=kept ||
           (STREAMING ? (requests<kept || requests>kept+1) : requests!=(kept==0?1:kept)) ||
           useful!=expected_useful || weight_bytes!=rows*kept || scan_frames!=passes*frames)
            $fatal(1,"traffic mismatch writes=%0d kept=%0d requests=%0d useful=%0d expected=%0d",writes,kept,requests,useful,expected_useful);
        $fdisplay(csv,"%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",name,BITMAP,STREAMING,length,rows,shift,passes,bw,kept,first_result-start_cycle,scan_cycles,scan_frames,scan_stalls,writes,requests,weight_bytes,useful,result_stalls);
        tick();
    endtask
    initial begin
        csv=$fopen($sformatf("activation_sharing_%0d_%0d.csv",BITMAP,STREAMING),"w");
        $fdisplay(csv,"fixture,bitmap,streaming,k,rows,shift,passes,activation_bw,kept,cycles,scan_cycles,scan_frames,scan_stalls,descriptor_writes,requests,weight_bytes,useful_macs,result_stalls");
        reset();
        for(integer d=0;d<6;d++) begin
            integer density;
            case(d) 0:density=1000;1:density=500;2:density=100;3:density=10;4:density=1;default:density=0;endcase
            for(integer supply=0;supply<2;supply++) begin
                prepare(4096,8,density,0,0);
                run_case($sformatf("density_%0d",density),1,supply==0?8:64,0,0);
                run_case($sformatf("density_%0d",density),8,supply==0?8:64,0,0);
            end
        end
        for(integer s=0;s<16;s++) begin
            prepare(65,3,1000,s,1); run_case("signed_threshold_stalls",1,8,1,0);
        end
        prepare(0,8,0,0,0); run_case("empty_vector",1,8,1,0);
        prepare(3,1,1000,0,1); run_case("short_row",1,64,1,0);
        prepare(1024,8,1000,0,0); run_case("reset_collect",1,64,0,1);
        prepare(128,8,1000,0,0); run_case("reset_replay",1,64,1,2);
        prepare(129,7,500,0,1); run_case("after_reset",1,64,1,0);
        $fclose(csv);
        $display("PASS: activation sharing BITMAP=%0d STREAMING=%0d vectors=%0d dots=%0d cycles=%0d",BITMAP,STREAMING,checked,total_dots,cycle);
        $finish;
    end
endmodule
