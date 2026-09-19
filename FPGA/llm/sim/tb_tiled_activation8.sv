`timescale 1ns/1ps
module tb_tiled_activation8 #(parameter integer CLOCK_MHZ=100, MEMORY_SCALE=1);
    logic clk=0;always #(500.0/CLOCK_MHZ) clk=~clk;
    logic rst_n=0,cmd_valid=0,cmd_ready;
    logic [6:0] cmd_rows,cmd_blocks;
    logic [3:0] cmd_shift;
    logic cmd_skip,cmd_reuse;
    logic [63:0] cmd_block_map;
    wire a_req_valid,a_req_last,a_rsp_ready;
    logic a_req_ready=0,a_rsp_valid=0,a_rsp_last=0;
    wire [5:0] a_req_block;
    logic [5:0] a_rsp_block=0;
    logic [511:0] a_rsp_values=0;
    logic [63:0] a_rsp_mask=0;
    wire req_valid,req_last,rsp_ready,result_valid,result_last;
    logic req_ready=0,rsp_valid=0,rsp_last=0,result_ready=1;
    wire [11:0] req_index;
    wire [2:0] req_tile,result_tile;
    wire [7:0] req_activation,req_mask,result_mask;
    logic [7:0] rsp_activation=0,rsp_mask=0;
    logic [63:0] rsp_weights=0;
    wire [255:0] result_values;
    wire debug_scanning,debug_selected,debug_cache_write,debug_cache_read;
    llm_tiled_activation8 dut(.*);
    logic p_cmd_valid=0,p_in_valid=0,p_last=0,p_out_ready=0;
    wire p_cmd_ready,p_in_ready,p_out_valid;
    logic [63:0] p_values=0;
    logic [7:0] p_mask=0;
    wire [63:0] p_blocks;
    llm_activation_block_map producer(.clk,.rst_n,.cmd_valid(p_cmd_valid),.cmd_ready(p_cmd_ready),
        .cmd_shift,.in_valid(p_in_valid),.in_ready(p_in_ready),.in_values(p_values),
        .in_mask(p_mask),.in_last(p_last),.out_valid(p_out_valid),.out_ready(p_out_ready),.out_blocks(p_blocks));
    integer x[0:4095],w[0:63][0:4095],perm[0:4095],indices[0:4095];
    logic [31:0] golden[0:63];
    logic [63:0] expected_map,actual_map;
    integer length,rows,kept,blocks,tiles,bw,producer_cycles;
    integer cycle=0,start_cycle,finish_cycle,checked=0,checked_tiles=0,checked_dots=0;
    integer activation_reads,activation_bytes,weight_bytes,selected,writes,reads,scan_cycles;
    integer expected_useful,useful,last_result_tile,request_pos[0:7],tile_stalls[0:7];
    integer source_cursor[0:7];
    integer aqh=0,aqt=0,aqn=0,adue[0:7],ablock[0:7],fetch_available=0;
    bit alast[0:7];
    integer wqh=0,wqt=0,wqn=0,wdue[0:7],weight_available=0;
    logic [7:0] wa[0:7],wm[0:7];logic [63:0] ww[0:7];bit wl[0:7];
    bit stalls=0,pending=0;
    bit hold_a=0,hold_w=0,hold_r=0;
    logic [7:0] old_a;
    logic [32:0] old_w;
    logic [268:0] old_r;
    integer accepted_cmd,accepted_result;
    logic [31:0] rng;
    integer csv;
    task automatic random_next;
        rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);
    endtask
    task automatic tick;
        integer n,active_rows,expected_block,t;
        logic [255:0] expected_vector;
        a_req_ready=aqn<8 && (!stalls || cycle%9!=0);
        a_rsp_valid=aqn>0 && adue[aqh]<=cycle;
        a_rsp_values=0;a_rsp_mask=0;a_rsp_block=6'(ablock[aqh]);a_rsp_last=alast[aqh];
        if(a_rsp_valid) for(integer i=0;i<64;i++) if(ablock[aqh]*64+i<length) begin
            a_rsp_values[i*8+:8]=8'(x[ablock[aqh]*64+i]);a_rsp_mask[i]=1;
        end
        req_ready=wqn<8 && (!stalls || cycle%7!=0);
        rsp_valid=wqn>0 && wdue[wqh]<=cycle;
        rsp_activation=wa[wqh];rsp_mask=wm[wqh];rsp_weights=ww[wqh];rsp_last=wl[wqh];
        result_ready=!stalls || tile_stalls[result_tile]>=5;
        @(posedge clk);
        accepted_cmd=rst_n && cmd_valid && cmd_ready;
        accepted_result=rst_n && result_valid && result_ready;
        if(rst_n) begin
            if(hold_a && {a_req_valid,a_req_block,a_req_last}!==old_a)$fatal(1,"activation request changed under stall");
            if(hold_w && {req_valid,req_index,req_tile,req_activation,req_mask,req_last}!==old_w)$fatal(1,"weight request changed under stall");
            if(hold_r && {result_valid,result_tile,result_mask,result_last,result_values}!==old_r)$fatal(1,"result changed under stall");
            if(accepted_cmd)begin start_cycle=cycle;pending=1;end
            if(debug_scanning)scan_cycles++;
            if(debug_selected)selected++;
            if(debug_cache_write)writes++;
            if(debug_cache_read)reads++;
            if(a_rsp_valid && a_rsp_ready)begin aqh=(aqh+1)%8;aqn--;end
            if(a_req_valid && a_req_ready)begin
                t=dut.tile;
                expected_block=source_cursor[t];
                while(expected_block<blocks && cmd_skip && !expected_map[expected_block])expected_block++;
                if(expected_block>=blocks || a_req_block!=expected_block || (cmd_reuse && t!=0))$fatal(1,"wrong/redundant activation read");
                source_cursor[t]=expected_block+1;
                expected_block++;
                while(expected_block<blocks && cmd_skip && !expected_map[expected_block])expected_block++;
                if(a_req_last!=(expected_block==blocks))$fatal(1,"activation last incorrect");
                n=length-a_req_block*64;if(n>64)n=64;
                activation_reads++;activation_bytes+=n;
                ablock[aqt]=a_req_block;alast[aqt]=a_req_last;
                // Contiguous local reads share a B-byte/cycle assembly port.
                // Eight finite slots include outstanding reads and held responses.
                if(fetch_available<cycle)fetch_available=cycle;
                fetch_available+=MEMORY_SCALE*((n+bw-1)/bw);
                adue[aqt]=fetch_available+2*MEMORY_SCALE;
                aqt=(aqt+1)%8;aqn++;
            end
            if(rsp_valid && rsp_ready)begin
                for(integer r=0;r<8;r++)if(rsp_mask[r] && rsp_activation!=0 && rsp_weights[r*8+:8]!=0)useful++;
                wqh=(wqh+1)%8;wqn--;
            end
            if(req_valid && req_ready)begin
                t=req_tile;active_rows=rows-t*8;if(active_rows>8)active_rows=8;
                if(t!=last_result_tile+1 || t>=tiles)$fatal(1,"wrong request tile");
                if(req_mask==0)begin
                    if(!req_last || request_pos[t]!=kept)$fatal(1,"bad empty marker");
                end else begin
                    if(request_pos[t]>=kept || req_index!=indices[request_pos[t]] || $signed(req_activation)!=x[req_index] ||
                       req_mask!=8'((1<<active_rows)-1) || (req_last && request_pos[t]!=kept-1))$fatal(1,"bad replay descriptor");
                    request_pos[t]++;
                end
                wa[wqt]=req_activation;wm[wqt]=req_mask;wl[wqt]=req_last;ww[wqt]=0;
                for(integer r=0;r<8;r++)if(req_mask[r])begin ww[wqt][r*8+:8]=8'(w[t*8+r][req_index]);weight_bytes++;end
                if(weight_available<cycle)weight_available=cycle;
                wdue[wqt]=weight_available+2*MEMORY_SCALE;
                weight_available+=MEMORY_SCALE;
                wqt=(wqt+1)%8;wqn++;
            end
            if(result_valid)begin
                expected_vector=0;t=result_tile;active_rows=rows-t*8;if(active_rows>8)active_rows=8;
                for(integer r=0;r<active_rows;r++)expected_vector[r*32+:32]=golden[t*8+r];
                if(!pending || t!=last_result_tile+1 || result_values!==expected_vector ||
                   result_mask!=8'((1<<active_rows)-1) || result_last!=(t==tiles-1) ||
                   request_pos[t]!=kept || aqn!=0 || wqn!=0)$fatal(1,"incorrect/early tile result");
                if(!result_ready)tile_stalls[t]++;
                if(accepted_result)begin
                    last_result_tile=t;checked_tiles++;checked_dots+=active_rows;
                    if(result_last)begin pending=0;finish_cycle=cycle;checked++;end
                end
            end
        end
        hold_a=rst_n && a_req_valid && !a_req_ready;old_a={a_req_valid,a_req_block,a_req_last};
        hold_w=rst_n && req_valid && !req_ready;old_w={req_valid,req_index,req_tile,req_activation,req_mask,req_last};
        hold_r=rst_n && result_valid && !result_ready;old_r={result_valid,result_tile,result_mask,result_last,result_values};
        cycle++;if(cycle>5000000)$fatal(1,"timeout");
        #1;@(negedge clk);
    endtask
    task automatic reset;
        rst_n=0;cmd_valid=0;p_cmd_valid=0;p_in_valid=0;p_out_ready=0;
        pending=0;aqn=0;aqh=0;aqt=0;wqn=0;wqh=0;wqt=0;weight_available=cycle;
        hold_a=0;hold_w=0;hold_r=0;repeat(4)tick();rst_n=1;
    endtask
    task automatic prepare(input integer n,nrows,density,xshift,special);
        integer j,tmp,v,beats,began;
        longint signed sum;
        length=n;rows=nrows;blocks=(n+63)/64;tiles=(nrows+7)/8;cmd_shift=4'(xshift);
        kept=0;expected_map=0;expected_useful=0;rng=32'h617A93B1;
        for(integer k=0;k<n;k++)perm[k]=k;
        for(integer k=n-1;k>0;k--)begin random_next();j=rng%(k+1);tmp=perm[k];perm[k]=perm[j];perm[j]=tmp;end
        for(integer k=0;k<n;k++)begin
            random_next();v=1+rng%127;if(rng[9])v=-v;
            x[k]=perm[k]<n*density/1000 ? v : 0;
            if(special==2)x[k]=k<n*density/1000 ? v : 0;
            if(special==1)x[k]=(k%5==0)?-128:(k%5==1)?-1:(k%5==2)?0:(k%5==3)?1:127;
            if((x[k]<0?-x[k]:x[k])>=(1<<xshift))begin indices[kept]=k;kept++;expected_map[k/64]=1;end
            for(integer r=0;r<64;r++)begin
                random_next();v=1+rng%127;if(rng[11])v=-v;
                if(special==1)v=(k+r)%16-8;
                w[r][k]=v;
            end
        end
        for(integer r=0;r<nrows;r++)begin
            sum=0;
            for(integer k=0;k<n;k++)if((x[k]<0?-x[k]:x[k])>=(1<<xshift))begin
                sum+=x[k]*w[r][k];if(x[k]!=0 && w[r][k]!=0)expected_useful++;
            end
            golden[r]=sum[31:0];
        end
        // Generate metadata in actual RTL, charge this pass separately.
        p_cmd_valid=1;began=cycle;tick();if(!p_cmd_ready && !producer.active)$fatal(1,"producer command");p_cmd_valid=0;
        beats=n==0?1:(n+7)/8;
        for(integer b=0;b<beats;b++)begin
            p_values=0;p_mask=0;
            for(integer i=0;i<8;i++)if(b*8+i<n)begin p_values[i*8+:8]=8'(x[b*8+i]);p_mask[i]=1;end
            p_in_valid=1;p_last=b==beats-1;tick();
        end
        p_in_valid=0;
        while(!p_out_valid)tick();
        producer_cycles=cycle-began;
        actual_map=p_blocks;if(actual_map!==expected_map)$fatal(1,"producer bitmap incorrect");
        repeat(3)begin tick();if(!p_out_valid || p_blocks!==actual_map)$fatal(1,"producer stalled metadata changed");end
        p_out_ready=1;tick();p_out_ready=0;
    endtask
    task automatic run_case(input string name,input bit skip,reuse,input integer supply,input bit stressed,input integer abort_at);
        integer passes,expected_bytes,expected_blocks,n;
        stalls=stressed;bw=supply;activation_reads=0;activation_bytes=0;weight_bytes=0;
        selected=0;writes=0;reads=0;scan_cycles=0;useful=0;last_result_tile=-1;fetch_available=cycle;
        for(integer t=0;t<8;t++)begin request_pos[t]=0;source_cursor[t]=0;tile_stalls[t]=0;end
        cmd_rows=7'(rows);cmd_blocks=7'(blocks);cmd_skip=skip;cmd_reuse=reuse;cmd_block_map=actual_map;
        // One 8-byte metadata read at the same 8-byte supply rate, before command.
        if(skip)tick();
        cmd_valid=1;do tick();while(!accepted_cmd);cmd_valid=0;
        while(pending)begin
            tick();
            if((abort_at==1 && selected>=3) || (abort_at==2 && last_result_tile==0 && reads>=3))begin reset();return;end
        end
        passes=reuse?1:tiles;expected_bytes=0;expected_blocks=0;
        for(integer b=0;b<blocks;b++)if(!skip || expected_map[b])begin
            n=length-b*64;if(n>64)n=64;expected_bytes+=n;expected_blocks++;
        end
        if(activation_reads!=passes*expected_blocks || activation_bytes!=passes*expected_bytes ||
           weight_bytes!=rows*kept || selected!=passes*kept || writes!=(reuse?kept:0) ||
           reads!=(reuse?(tiles-1)*kept:0) || useful!=expected_useful)$fatal(1,"traffic counts incorrect");
        $fdisplay(csv,"%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",name,length,rows,cmd_shift,skip,reuse,bw,kept,$countones(expected_map),finish_cycle-start_cycle+skip,producer_cycles,activation_reads,activation_bytes,skip?8:0,weight_bytes,selected,writes,reads,scan_cycles,useful);
        tick();
    endtask
    task automatic producer_exhaustive;
        integer v;
        logic [63:0] expected;
        for(integer s=0;s<16;s++)for(integer sample=0;sample<256;sample++)for(integer lane=0;lane<8;lane++)begin
            v=sample<128 ? sample : sample-256;
            expected=(v<0?-v:v)>=(1<<s) ? 64'd1 : 64'd0;
            cmd_shift=4'(s);p_cmd_valid=1;tick();p_cmd_valid=0;
            p_values=0;p_values[lane*8+:8]=8'(sample);p_mask=8'd1<<lane;
            p_in_valid=1;p_last=1;tick();p_in_valid=0;
            // Magnitude and threshold pipeline stages precede bitmap commit.
            if(p_out_valid)$fatal(1,"producer result arrived before pipeline drain");
            tick();if(p_out_valid)$fatal(1,"producer result arrived one cycle early");
            tick();
            if(!p_out_valid || p_blocks!==expected)$fatal(1,"producer singleton threshold mismatch");
            p_out_ready=1;tick();p_out_ready=0;
        end
        $display("PASS: producer exhaustive cases=32768");
    endtask
    initial begin
        csv=$fopen("tiled_activation.csv","w");
        $fdisplay(csv,"fixture,k,rows,shift,skip,reuse,activation_bw,kept,occupied_blocks,consumer_cycles,producer_cycles,activation_requests,activation_bytes,metadata_read_bytes,weight_bytes,selected,cache_writes,cache_reads,scan_cycles,useful_macs");
        reset();
        producer_exhaustive();
        for(integer d=0;d<6;d++)begin
            integer density;
            case(d)0:density=1000;1:density=500;2:density=100;3:density=10;4:density=1;default:density=0;endcase
            for(integer r=0;r<2;r++)begin
                prepare(4096,r==0?8:64,density,0,0);
                for(integer mode=0;mode<4;mode++)run_case($sformatf("density_%0d",density),mode[0],mode[1],8,0,0);
            end
        end
        prepare(4096,64,1,0,0);for(integer mode=0;mode<4;mode++)run_case("wide_supply",mode[0],mode[1],64,0,0);
        prepare(4096,64,10,0,2);for(integer mode=0;mode<4;mode++)run_case("clustered_one_percent",mode[0],mode[1],8,0,0);
        for(integer s=0;s<16;s++)begin
            prepare(129,17,1000,s,1);
            run_case("signed_threshold_stalls",1,1,8,1,0);
        end
        prepare(0,9,0,0,0);run_case("empty_vector",1,1,8,1,0);
        prepare(3,1,1000,0,1);run_case("short_row",1,1,8,1,0);
        prepare(1024,17,1000,0,0);run_case("reset_first_tile",1,1,8,1,1);
        prepare(128,17,1000,0,0);run_case("reset_replay",1,1,8,1,2);
        // Cancel a nonempty, incomplete producer vector; the next map must be zero.
        cmd_shift=0;p_cmd_valid=1;tick();p_cmd_valid=0;
        p_values='1;p_mask='1;p_last=0;p_in_valid=1;tick();reset();
        prepare(65,9,0,0,0);run_case("producer_reset",1,1,8,1,0);
        prepare(65,9,500,0,1);run_case("after_reset",1,1,8,1,0);
        $fclose(csv);
        $display("PASS: tiled activation commands=%0d tiles=%0d dots=%0d cycles=%0d",checked,checked_tiles,checked_dots,cycle);
        $finish;
    end
endmodule
