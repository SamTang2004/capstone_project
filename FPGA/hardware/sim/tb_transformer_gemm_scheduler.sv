module tb_transformer_gemm_scheduler;
    logic clk=0, rst_n=0, pipeline_enable=1, start=0, weight_int4=0, fused_requant=0;
    logic [1:0] fused_output_mode=0;
    logic [31:0] a_base=32'h100,b_base=32'h400,c_base=32'h800,c_stride=8;
    logic [31:0] bias_base=0;
    logic signed [31:0] rq_multiplier=1,rq_output_zero_point=0;
    logic [5:0] rq_right_shift=0;
    logic [15:0] dim_m=3,dim_n=2,dim_k=2;
    logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
    logic [31:0] debug_state;
    logic [63:0] phase_load_a_cycles,phase_load_b_cycles,phase_compute_cycles,phase_write_cycles;
    logic [63:0] phase_load_b_weight_cycles,phase_load_b_bias_cycles;
    logic [63:0] w4_cache_hit_tiles,w4_cache_miss_tiles;
    logic read_req_valid,read_req_ready,read_req_wide,read_req_qword,read_req_allow_ahead;
    logic [31:0] read_req_addr; logic [6:0] read_req_len;
    logic read_qword_valid,read_qword_ready,read_qword_last;
    logic [63:0] read_qword_data; logic [7:0] read_qword_keep;
    logic write_req_valid,write_req_ready=1,write_req_last,memory_busy=0,memory_error=0;
    logic [31:0] write_req_addr,write_req_data; logic [3:0] write_req_strb;
    logic [7:0] mem[0:4095];
    logic pending; logic [31:0] pending_addr; logic [6:0] pending_len;
    integer errors=0, read_requests=0;
    always #5 clk=~clk;

    transformer_gemm_scheduler #(
        .MAX_K(16), .MAX_N(16), .MAX_W4_CACHE_VECTORS(32)) dut(.*);
    assign read_req_ready = !pending && !read_qword_valid;

    always_ff @(posedge clk) begin
        if(!rst_n) begin pending<=0;read_qword_valid<=0;read_qword_data<=0;read_qword_keep<=0;read_qword_last<=0;read_requests<=0; end
        else begin
            if(read_req_valid&&read_req_ready) begin pending<=1;pending_addr<=read_req_addr;pending_len<=read_req_len;read_requests<=read_requests+1; end
            if(pending&&!read_qword_valid) begin
                read_qword_data<=0;read_qword_keep<=0;
                for(int i=0;i<8;i++) if(i<pending_len) begin read_qword_data[i*8+:8]<=mem[pending_addr+i];read_qword_keep[i]<=1;end
                read_qword_last<=pending_len<=8;read_qword_valid<=1;
            end else if(read_qword_valid&&read_qword_ready) begin
                read_qword_valid<=0;read_qword_last<=0;
                if(read_qword_last) pending<=0;
                else begin pending_addr<=pending_addr+8;pending_len<=pending_len-8;end
            end
            if(write_req_valid&&write_req_ready)
                for(int i=0;i<4;i++) if(write_req_strb[i]) mem[write_req_addr+i]<=write_req_data[i*8+:8];
        end
    end

    task automatic put_qword(input int addr,input logic [63:0] value);
        for(int i=0;i<8;i++)mem[addr+i]=value[i*8+:8];
    endtask
    task automatic put_a_row_vector(input int addr,input int first_row);
        logic [63:0] value;
        begin
            value = '0;
            for(int lane=0;lane<8;lane++)
                if(first_row+lane<20)value[lane*8+:8]=first_row+lane+1;
            put_qword(addr,value);
        end
    endtask
    task automatic put_w4_vector(input int addr,input int first_col,input int k_index);
        logic [31:0] value;
        integer weight;
        begin
            value = '0;
            for(int col=0;col<8;col++)begin
                weight=((first_col+col+k_index)%5)-2;
                value[col*4+:4]=weight[3:0];
            end
            for(int byte_index=0;byte_index<4;byte_index++)
                mem[addr+byte_index]=value[byte_index*8+:8];
        end
    endtask
    task automatic expect_i32(input int addr,input integer expected);
        logic signed [31:0] got;
        begin got={mem[addr+3],mem[addr+2],mem[addr+1],mem[addr]};if(got!==expected)begin $error("addr=%h expected=%0d got=%0d",addr,expected,got);errors++;end end
    endtask
    task automatic expect_i8(input int addr,input integer expected);
        logic signed [7:0] got;
        begin got=mem[addr];if(got!==expected)begin $error("addr=%h expected=%0d got=%0d",addr,expected,got);errors++;end end
    endtask
    task automatic launch;
        begin @(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(posedge clk); end
    endtask

    initial begin
        for(int i=0;i<4096;i++)mem[i]=0;
        repeat(4)@(posedge clk);rst_n=1;
        // A is tile-major Kx8: [column k][eight output rows].
        put_qword('h100,64'h0000000000050301); put_qword('h108,64'h0000000000060402);
        put_qword('h400,64'h0000000000000807); put_qword('h408,64'h0000000000000a09);
        launch();
        expect_i32('h800,25);expect_i32('h804,28);expect_i32('h808,57);
        expect_i32('h80c,64);expect_i32('h810,89);expect_i32('h814,100);
        if(read_requests!=2)begin $error("W8 line requests expected=2 got=%0d",read_requests);errors++;end

        // Two M tiles execute concurrently and share one B-tile load.
        dim_m=10;dim_n=2;dim_k=2;c_stride=8;
        put_qword('h100,64'h0807060504030201);put_qword('h108,0);
        put_qword('h110,64'h0000000000000a09);put_qword('h118,0);
        put_qword('h400,64'h0000000000000302);put_qword('h408,0);
        for(int i='h800;i<'h850;i++)mem[i]=0;
        launch();
        for(int row=0;row<10;row++)begin
            expect_i32('h800+row*8,(row+1)*2);
            expect_i32('h804+row*8,(row+1)*3);
        end
        if(read_requests!=5)begin $error("dual-core W8 expected 3 additional requests, cumulative=%0d",read_requests);errors++;end

        for(int i='h800;i<'h818;i++)mem[i]=0;
        dim_m=3;dim_n=2;dim_k=2;c_stride=8;weight_int4=1;
        put_qword('h100,64'h0000000000050301); put_qword('h108,64'h0000000000060402);
        // W4 vectors: k0=[-2,3], k1=[4,-1].
        mem['h400]=8'h3e;mem['h401]=0;mem['h402]=0;mem['h403]=0;
        mem['h404]=8'hf4;mem['h405]=0;mem['h406]=0;mem['h407]=0;
        launch();
        expect_i32('h800,6);expect_i32('h804,1);expect_i32('h808,10);
        expect_i32('h80c,5);expect_i32('h810,14);expect_i32('h814,9);
        if(read_requests!=7)begin $error("wide W4 fill expected one A and one B request, cumulative=%0d",read_requests);errors++;end

        // A 20-row W4 GEMM needs two M-pair passes. The second pass must hit
        // the BRAM cache and therefore adds only one A request, not two more B
        // vector requests. Total: three A requests plus two initial B fills.
        dim_m=20;dim_n=2;dim_k=2;c_stride=8;
        put_qword('h100,64'h0807060504030201);put_qword('h108,0);
        put_qword('h110,64'h100f0e0d0c0b0a09);put_qword('h118,0);
        put_qword('h120,64'h0000000014131211);put_qword('h128,0);
        for(int i='h800;i<'h8a0;i++)mem[i]=0;
        launch();
        for(int row=0;row<20;row++)begin
            expect_i32('h800+row*8,(row+1)*-2);
            expect_i32('h804+row*8,(row+1)*3);
        end
        if(read_requests!=11)begin
            $error("wide W4 BRAM reuse expected 4 additional requests, cumulative=%0d",read_requests);errors++;
        end
        if(!phase_load_a_cycles||!phase_load_b_cycles||!phase_compute_cycles||!phase_write_cycles)begin
            $error("Missing GEMM phase cycles A=%0d B=%0d compute=%0d write=%0d",phase_load_a_cycles,phase_load_b_cycles,phase_compute_cycles,phase_write_cycles);errors++;
        end
        if(!phase_load_b_weight_cycles||phase_load_b_cycles < phase_load_b_weight_cycles||
           !w4_cache_hit_tiles||!w4_cache_miss_tiles)begin
            $error("Missing detailed B counters weight=%0d bias=%0d hits=%0d misses=%0d",
                phase_load_b_weight_cycles,phase_load_b_bias_cycles,
                w4_cache_hit_tiles,w4_cache_miss_tiles);errors++;
        end

        // Fuse bias and requantization into GEMM so no INT32 accumulator tile
        // is written to DDR. Output is directly packed as tile-major INT8.
        dim_m=3;dim_n=2;dim_k=2;c_base=32'h900;c_stride=0;
        weight_int4=1;fused_requant=1;bias_base=32'h600;
        rq_multiplier=1;rq_right_shift=0;rq_output_zero_point=0;
        put_qword('h100,64'h0000000000050301);put_qword('h108,64'h0000000000060402);
        put_qword('h600,64'hfffffffe0000000a);
        for(int i='h900;i<'h940;i++)mem[i]=0;
        launch();
        expect_i8('h900,16);expect_i8('h901,20);expect_i8('h902,24);
        expect_i8('h908,-1);expect_i8('h909,3);expect_i8('h90a,7);
        for(int row=3;row<8;row++)begin
            expect_i8('h900+row,0);expect_i8('h908+row,0);
        end
        if(read_requests!=14)begin
            $error("Fused W4A8 expected 3 additional requests, cumulative=%0d",read_requests);errors++;
        end
        // The same fused tile can write dense row-major INT8 without an INT32
        // round trip. Rows are separated by the logical N-byte stride.
        fused_output_mode=1;c_base=32'h980;c_stride=2;
        launch();
        expect_i8('h980,16);expect_i8('h981,-1);
        expect_i8('h982,20);expect_i8('h983,3);
        expect_i8('h984,24);expect_i8('h985,7);
        // Direct dynamic-B output emits one padded 8-byte vector per M row.
        fused_output_mode=2;c_base=32'ha00;c_stride=8;
        launch();
        expect_i8('ha00,16);expect_i8('ha01,-1);
        expect_i8('ha08,20);expect_i8('ha09,3);
        expect_i8('ha10,24);expect_i8('ha11,7);
        if(read_requests!=20)begin
            $error("Fused output modes expected 6 additional requests, cumulative=%0d",read_requests);errors++;
        end

        // Patch embedding walks multiple N tiles, then reuses every cached W4
        // tile for the next 16-row M pass while prior fused writes drain.  Use
        // K=4 so both halves of each cached 64-bit qword are consumed.
        dim_m=20;dim_n=16;dim_k=4;
        a_base=32'h100;b_base=32'h400;c_base=32'hb00;c_stride=16;
        bias_base=32'h600;weight_int4=1;fused_requant=1;fused_output_mode=1;
        rq_multiplier=1;rq_right_shift=0;rq_output_zero_point=0;
        for(int col=0;col<16;col++)
            for(int byte_index=0;byte_index<4;byte_index++)
                mem['h600+col*4+byte_index]=(32'd1 >> (byte_index*8));
        for(int group=0;group<20;group+=16)
            for(int bank=0;bank<2;bank++)
                for(int k_index=0;k_index<4;k_index++)
                    put_a_row_vector('h100+group*4+bank*32+k_index*8,
                                     group+bank*8);
        for(int first_col=0;first_col<16;first_col+=8)
            for(int k_index=0;k_index<4;k_index++)
                put_w4_vector('h400+first_col*2+k_index*4,first_col,k_index);
        for(int addr='hb00;addr<'hc40;addr++)mem[addr]=0;
        launch();
        for(int row=0;row<20;row++)begin
            for(int col=0;col<16;col++)begin
                integer expected;
                expected=1;
                for(int k_index=0;k_index<4;k_index++)
                    expected=expected+(row+1)*(((col+k_index)%5)-2);
                if(expected>127)expected=127;
                if(expected< -128)expected=-128;
                expect_i8('hb00+row*16+col,expected);
            end
        end
        // Two bias tiles are fetched only for the first M pass.  The second
        // pass reuses both 256-bit cached vectors: 3 A + 2 B + 8 bias reads.
        if(read_requests!=33)begin
            $error("W4 prefetch/bias-cache expected 13 additional requests, cumulative=%0d",read_requests);errors++;
        end
        if(error||errors)$fatal(1,"Transformer scheduler failed errors=%0d",errors);
        $display("Transformer tiled GEMM scheduler golden trace passed");$finish;
    end
    initial begin #200000;$fatal(1,
        "Scheduler timeout state=%0d m=%0d n=%0d pf=%0d pf_issue=%0d pf_recv=%0d fw_busy=%0d fw_capture=%0d fw_done=%0d fw_error=%0d wr_valid=%0d",
        dut.state,dut.m_index,dut.n_index,dut.prefetch_active,dut.prefetch_issue_index,dut.prefetch_recv_index,
        dut.fused_writer_busy,dut.fused_writer_capture_done,dut.fused_writer_done,
        dut.fused_writer_error,write_req_valid);end
endmodule
