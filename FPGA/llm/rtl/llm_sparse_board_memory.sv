// Board-only on-chip memory harness. CPU uploads are outside measured execution.
// Activation reads assemble 64-byte frames; eight weight banks sustain one column/cycle.
module llm_sparse_board_memory (
    input logic clk,rst_n,
    input logic [31:0] ctrl,
    input logic [511:0] staging,
    output logic [31:0] status,
    output logic [511:0] perf
);
    typedef enum logic [2:0] {IDLE,PCMD,PFEED,PWAIT,CCMD,RUN} state_t;
    state_t state;
    logic [3:0] previous;
    wire [3:0] event_bits=ctrl[3:0]^previous;
    wire cancel=event_bits[3];
    // Register cancellation before distributing it across the complete core.
    // Synchronous reset/cancel takes effect one clock after it is sampled.
    logic engine_reset=0;
    always_ff @(posedge clk)engine_reset<=rst_n && !cancel;
    wire [12:0] cfg_k=staging[2*32+:13];
    wire [31:0] cfg=staging[3*32+:32];
    wire [15:0] upload_address=staging[4*32+:16];
    wire [31:0] upload_data=staging[5*32+:32];
    logic done,error;
    logic [12:0] k;
    logic [6:0] rows,blocks;
    logic [3:0] shift;
    logic skip,reuse;
    logic [63:0] block_map;
    logic [31:0] cycles,prep_cycles,activation_requests,weight_requests,scan_cycles;
    logic [31:0] results[0:63];
    logic [31:0] result_read;
    (* ram_style="block" *) logic [31:0] activation_low[0:511],activation_high[0:511];
    logic [31:0] a_low_q,a_high_q;
    logic a_read;
    logic [8:0] a_address;
    logic a_read_q,issue_last;
    logic [8:0] a_address_q;
    logic [2:0] issue_position;
    wire write_a=rst_n && state==IDLE && event_bits[1];
    wire write_w=rst_n && state==IDLE && event_bits[2];
    always_ff @(posedge clk) begin
        if(write_a && !upload_address[0])activation_low[upload_address[9:1]]<=upload_data;
        if(write_a && upload_address[0])activation_high[upload_address[9:1]]<=upload_data;
        if(a_read_q)begin a_low_q<=activation_low[a_address_q];a_high_q<=activation_high[a_address_q];end
        result_read<=results[staging[6*32+:6]];
    end
    wire [63:0] a_data={a_high_q,a_low_q};
    logic [63:0] a_data_pipe;
    logic memory_read_valid,memory_read_last;
    logic [7:0] memory_read_mask;
    logic [2:0] memory_read_position;
    logic read_valid,read_last;
    logic [7:0] read_mask;
    logic [2:0] read_position;
    logic [9:0] producer_issue,producer_beats;
    wire p_cmd_ready,p_in_ready,p_out_valid;
    wire [63:0] p_blocks;
    llm_activation_block_map producer(
        .clk,.rst_n(engine_reset),.cmd_valid(state==PCMD),.cmd_ready(p_cmd_ready),.cmd_shift(shift),
        .in_valid(state==PFEED && read_valid),.in_ready(p_in_ready),.in_values(a_data_pipe),
        .in_mask(read_mask),.in_last(read_last),.out_valid(p_out_valid),.out_ready(state==PWAIT),.out_blocks(p_blocks)
    );
    wire cmd_ready,a_req_valid,a_req_last,a_rsp_ready,req_valid,req_last,rsp_ready;
    wire [5:0] a_req_block;
    wire [11:0] req_index;
    wire [2:0] req_tile,result_tile;
    wire [7:0] req_activation,req_mask,result_mask;
    wire result_valid,result_last;
    wire [255:0] result_values;
    wire debug_scanning,debug_selected,debug_cache_write,debug_cache_read;
    logic a_busy,a_rsp_valid,a_rsp_last;
    logic [5:0] a_block;
    logic [3:0] a_issue;
    logic [511:0] a_frame;
    logic [63:0] a_frame_mask;
    wire a_req_ready=state==RUN && !a_busy && !a_rsp_valid;
    // Fixed-latency memory pipeline with reserved response capacity. Credits
    // break the combinational ready chain from the core back into BRAM enables.
    logic address_valid,w_valid,word_valid,rsp_valid;
    wire rsp_space=!rsp_valid || rsp_ready;
    logic [4:0] inflight,queue_count;
    logic [3:0] queue_write,queue_read;
    (* ram_style="distributed" *) logic [80:0] response_queue[0:15];
    wire req_ready=inflight<12;
    wire dequeue=queue_count!=0 && rsp_space;
    wire [63:0] selected_weights;
    logic [12:0] weight_address;
    logic [1:0] address_byte,word_byte;
    logic [7:0] address_activation,address_mask,word_activation,word_mask;
    logic address_last,word_last;
    logic [31:0] registered_words[0:7];
    logic [1:0] w_byte;
    logic [7:0] w_activation,w_mask,rsp_activation,rsp_mask;
    logic w_last,rsp_last;
    logic [63:0] rsp_weights;
    for(genvar bank=0;bank<8;bank++)begin:weights
        (* ram_style="block" *) logic [31:0] memory[0:8191];
        logic [31:0] word_q;
        logic upload_enable;
        logic [12:0] local_upload_address;
        logic [31:0] local_upload_data;
        always_ff @(posedge clk)begin
            if(!rst_n)upload_enable<=0;
            else upload_enable<=write_w && upload_address[12:10]==bank;
            if(write_w && upload_address[12:10]==bank)begin
                local_upload_address<={upload_address[15:13],upload_address[9:0]};
                local_upload_data<=upload_data;
            end
            if(upload_enable)memory[local_upload_address]<=local_upload_data;
            if(address_valid)word_q<=memory[weight_address];
            if(w_valid)registered_words[bank]<=word_q;
        end
        assign selected_weights[bank*8+:8]=registered_words[bank][word_byte*8+:8];
    end
    always_ff @(posedge clk)begin
        if(word_valid)response_queue[queue_write]<={word_last,word_mask,word_activation,selected_weights};
    end
    llm_tiled_activation8 engine(
        .clk,.rst_n(engine_reset),.cmd_valid(state==CCMD),.cmd_ready,
        .cmd_rows(rows),.cmd_blocks(blocks),.cmd_shift(shift),.cmd_skip(skip),.cmd_reuse(reuse),.cmd_block_map(block_map),
        .a_req_valid,.a_req_ready,.a_req_block,.a_req_last,
        .a_rsp_valid,.a_rsp_ready,.a_rsp_block(a_block),.a_rsp_last,
        .a_rsp_values(a_frame),.a_rsp_mask(a_frame_mask),
        .req_valid,.req_ready,.req_index,.req_tile,.req_activation,.req_mask,.req_last,
        .rsp_valid,.rsp_ready,.rsp_activation,.rsp_mask,.rsp_weights,.rsp_last,
        .result_valid,.result_ready(1'b1),.result_values,.result_tile,.result_mask,.result_last,
        .debug_scanning,.debug_selected,.debug_cache_write,.debug_cache_read
    );
    always_comb begin
        a_read=0;a_address=0;
        if(state==PFEED && producer_issue<producer_beats)begin a_read=1;a_address=producer_issue[8:0];end
        else if(state==RUN && a_busy && a_issue<8)begin a_read=1;a_address={a_block,a_issue[2:0]};end
        status={29'b0,error,done,(state!=IDLE)};
        perf=0;perf[0+:32]=32'h53505202;
        perf[32+:32]=cycles;perf[64+:32]=prep_cycles;perf[96+:32]=activation_requests;
        perf[128+:32]=weight_requests;perf[160+:32]=result_read;
        perf[192+:64]=block_map;perf[256+:32]=scan_cycles;
    end
    always_ff @(posedge clk) begin
        if(!rst_n)previous<=0;else previous<=ctrl[3:0];
        if(!engine_reset)begin
            state<=IDLE;done<=0;error<=0;k<=0;rows<=0;blocks<=0;shift<=0;skip<=0;reuse<=0;
            block_map<=0;cycles<=0;prep_cycles<=0;activation_requests<=0;weight_requests<=0;scan_cycles<=0;
            read_valid<=0;read_last<=0;read_mask<=0;read_position<=0;producer_issue<=0;producer_beats<=0;
            memory_read_valid<=0;memory_read_last<=0;memory_read_mask<=0;memory_read_position<=0;
            a_read_q<=0;a_address_q<=0;issue_last<=0;issue_position<=0;
            a_busy<=0;a_rsp_valid<=0;a_rsp_last<=0;a_block<=0;a_issue<=0;a_frame<=0;a_frame_mask<=0;
            w_valid<=0;w_byte<=0;w_activation<=0;w_mask<=0;w_last<=0;
            address_valid<=0;weight_address<=0;address_byte<=0;address_activation<=0;address_mask<=0;address_last<=0;
            word_valid<=0;word_byte<=0;word_activation<=0;word_mask<=0;word_last<=0;
            inflight<=0;queue_count<=0;queue_write<=0;queue_read<=0;
            rsp_valid<=0;rsp_activation<=0;rsp_mask<=0;rsp_weights<=0;rsp_last<=0;
            for(integer i=0;i<64;i++)results[i]<=0;
        end else begin
            if(state!=IDLE && (|event_bits[2:0]))error<=1;
            a_read_q<=a_read;
            memory_read_valid<=a_read_q;
            read_valid<=memory_read_valid;
            if(memory_read_valid)begin
                a_data_pipe<=a_data;read_last<=memory_read_last;
                read_position<=memory_read_position;read_mask<=memory_read_mask;
            end
            if(a_read)begin
                a_address_q<=a_address;issue_last<=producer_issue+1'b1==producer_beats;
                issue_position<=a_issue[2:0];
                if(state==PFEED)producer_issue<=producer_issue+1'b1;
                else a_issue<=a_issue+1'b1;
            end
            if(a_read_q)begin
                memory_read_last<=issue_last;memory_read_position<=issue_position;
                for(integer i=0;i<8;i++)memory_read_mask[i]<={1'b0,a_address_q,3'(i)}<k;
            end
            if(a_req_valid && a_req_ready)begin
                a_busy<=1;a_block<=a_req_block;a_rsp_last<=a_req_last;a_issue<=0;activation_requests<=activation_requests+1'b1;
            end
            if(state==RUN && read_valid)begin
                a_frame[read_position*64+:64]<=a_data_pipe;
                a_frame_mask[read_position*8+:8]<=read_mask;
                if(read_position==7)begin a_busy<=0;a_rsp_valid<=1;end
            end
            if(a_rsp_valid && a_rsp_ready)a_rsp_valid<=0;
            begin
                address_valid<=req_valid && req_ready;
                if(req_valid && req_ready)begin
                    weight_address<={req_tile,req_index[11:2]};address_byte<=req_index[1:0];
                    address_activation<=req_activation;address_mask<=req_mask;address_last<=req_last;
                    weight_requests<=weight_requests+1'b1;
                end
            end
            begin
                w_valid<=address_valid;
                if(address_valid)begin
                    w_byte<=address_byte;w_activation<=address_activation;w_mask<=address_mask;w_last<=address_last;
                end
            end
            begin
                word_valid<=w_valid;
                if(w_valid)begin
                    word_byte<=w_byte;word_activation<=w_activation;word_mask<=w_mask;word_last<=w_last;
                end
            end
            if(rsp_space)begin
                rsp_valid<=dequeue;
                if(dequeue){rsp_last,rsp_mask,rsp_activation,rsp_weights}<=response_queue[queue_read];
            end
            if(word_valid)queue_write<=queue_write+1'b1;
            if(dequeue)queue_read<=queue_read+1'b1;
            case({word_valid,dequeue})
                2'b10:queue_count<=queue_count+1'b1;
                2'b01:queue_count<=queue_count-1'b1;
                default:begin end
            endcase
            case({req_valid && req_ready,rsp_valid && rsp_ready})
                2'b10:inflight<=inflight+1'b1;
                2'b01:inflight<=inflight-1'b1;
                default:begin end
            endcase
            if(state==PCMD || state==PFEED || state==PWAIT)prep_cycles<=prep_cycles+1'b1;
            if(state==RUN)begin
                cycles<=cycles+1'b1;
                if(debug_scanning)scan_cycles<=scan_cycles+1'b1;
            end
            if(result_valid)begin
                for(integer r=0;r<8;r++)results[{result_tile,3'(r)}]<=result_values[r*32+:32];
                if(result_last)begin state<=IDLE;done<=1;end
            end
            case(state)
                IDLE: if(event_bits[0])begin
                    if(cfg_k>4096 || cfg[6:0]==0 || cfg[6:0]>64)begin error<=1;done<=0;end
                    else begin
                        k<=cfg_k;rows<=cfg[6:0];blocks<=7'((cfg_k+13'd63)>>6);
                        shift<=cfg[11:8];skip<=cfg[16];reuse<=cfg[17];
                        producer_beats<=cfg_k==0 ? 10'd1 : 10'((cfg_k+13'd7)>>3);
                        producer_issue<=0;cycles<=0;prep_cycles<=0;activation_requests<=0;weight_requests<=0;scan_cycles<=0;
                        done<=0;error<=0;state<=PCMD;
                    end
                end
                PCMD: if(p_cmd_ready)state<=PFEED;
                PFEED: if(read_valid && read_last && p_in_ready)state<=PWAIT;
                PWAIT: if(p_out_valid)begin block_map<=p_blocks;state<=CCMD;end
                CCMD: if(cmd_ready)state<=RUN;
                default:begin end
            endcase
        end
    end
endmodule
