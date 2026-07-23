module tb_conv2d_tile_loader_ddr;
    logic clk=0,rst_n=0,start,matrix_select; logic [15:0] m0_cfg, k0_cfg, n0_cfg; logic [31:0] ra; logic [4:0] rl; logic rv,rr,rdv,rdr,rdlast,busy,done,error,cwe,cm; logic [7:0] rd,ca,cd; logic [7:0] mem[0:127], captured[0:31], weights_captured[0:31]; integer errors,cycles,start_cycle,first_request_cycle,request_count,request_count_start,max_request_len;
    logic read_pending;
    logic [31:0] read_address_q;
    logic [4:0] read_length_q, read_index_q;
    always #5 clk=~clk;
    conv2d_tile_loader_ddr dut(.clk(clk),.rst_n(rst_n),.start(start),.matrix_select(matrix_select),.conv_mode(1'b1),.base_addr(0),.row_stride(0),.input_base_addr(0),.weight_base_addr(64),.input_c(1),.input_h(3),.input_w(3),.out_c(1),.out_h(2),.out_w(2),.kernel_h(2),.kernel_w(2),.stride_h(1),.stride_w(1),.pad_h(0),.pad_w(0),.m0(m0_cfg),.k0(k0_cfg),.n0(n0_cfg),.tile_rows(2),.tile_cols(4),.busy(busy),.done(done),.error(error),.read_req_valid(rv),.read_req_ready(rr),.read_req_addr(ra),.read_req_len(rl),.read_data_valid(rdv),.read_data_ready(rdr),.read_data(rd),.read_data_last(rdlast),.cache_we(cwe),.cache_matrix(cm),.cache_addr(ca),.cache_wdata(cd));
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (start) begin start_cycle <= cycles; first_request_cycle <= -1; request_count_start <= request_count; max_request_len <= 0; end
        if (rv && rr) begin
            if(first_request_cycle < 0) first_request_cycle <= cycles;
            request_count <= request_count + 1;
            if(rl > max_request_len) max_request_len <= rl;
        end
        if (!rst_n) begin
            rr <= 0; rdv <= 0; rdlast <= 0; rd <= 0; read_pending <= 0; read_address_q <= 0; read_length_q <= 0; read_index_q <= 0;
        end else begin
            rr <= !read_pending;
            rdv <= 0; rdlast <= 0;
            if (rv && rr) begin
                read_pending <= 1;
                read_address_q <= ra;
                read_length_q <= rl;
                read_index_q <= 0;
            end
            if (read_pending && rdr) begin
                rdv <= 1;
                rd <= mem[read_address_q + read_index_q];
                rdlast <= (read_index_q + 1 == read_length_q);
                if(read_index_q + 1 == read_length_q) read_pending <= 0;
                else read_index_q <= read_index_q + 1'b1;
            end
            if (cwe) begin if(cm) weights_captured[ca] <= cd; else captured[ca] <= cd; end
        end
    end
    initial begin errors=0;cycles=0;start_cycle=-1;first_request_cycle=-1;request_count=0;request_count_start=0;max_request_len=0;start=0;matrix_select=0;m0_cfg=0;k0_cfg=0;n0_cfg=0;rr=0;rdv=0;rdlast=0;rd=0;read_pending=0;read_address_q=0;read_length_q=0;read_index_q=0; for(int i=0;i<9;i++)mem[i]=i+1; for(int i=0;i<4;i++)mem[64+i]=8'h40+i; for(int i=0;i<32;i++)begin captured[i]=0; weights_captured[i]=0; end repeat(3)@(posedge clk);rst_n=1;@(posedge clk);
        start=1;@(posedge clk);start=0;wait(done);#1;if(first_request_cycle-start_cycle!=22||cycles-start_cycle!=100)$fatal(1,"Conv input pipeline latency first=%0d done=%0d, expected 22/100",first_request_cycle-start_cycle,cycles-start_cycle);if(request_count-request_count_start!=4||max_request_len!=2)errors++;
        if(captured[0]!=1||captured[1]!=2||captured[2]!=4||captured[3]!=5||captured[16]!=2||captured[17]!=3||captured[18]!=5||captured[19]!=6)errors++;
        // Exercise consecutive tiles in both directions. Decode accumulators
        // must be initialized from each command, not retained from m0=2.
        m0_cfg=2;@(posedge clk);start=1;@(posedge clk);start=0;wait(done);#1;if(first_request_cycle-start_cycle!=23||cycles-start_cycle!=101)$fatal(1,"Offset Conv input pipeline latency first=%0d done=%0d, expected 23/101",first_request_cycle-start_cycle,cycles-start_cycle);
        if(captured[0]!=4||captured[1]!=5||captured[2]!=7||captured[3]!=8||captured[16]!=5||captured[17]!=6||captured[18]!=8||captured[19]!=9)errors++;
        m0_cfg=0;@(posedge clk);start=1;@(posedge clk);start=0;wait(done);#1;if(first_request_cycle-start_cycle!=22||cycles-start_cycle!=100)$fatal(1,"Repeated Conv input pipeline latency first=%0d done=%0d, expected 22/100",first_request_cycle-start_cycle,cycles-start_cycle);
        if(captured[0]!=1||captured[1]!=2||captured[2]!=4||captured[3]!=5||captured[16]!=2||captured[17]!=3||captured[18]!=5||captured[19]!=6)errors++;
        matrix_select=1;@(posedge clk);start=1;@(posedge clk);start=0;wait(done);#1;if(first_request_cycle-start_cycle!=28||cycles-start_cycle!=58)$fatal(1,"Conv weight pipeline latency first=%0d done=%0d, expected 28/58",first_request_cycle-start_cycle,cycles-start_cycle);if(request_count-request_count_start!=1||max_request_len!=2)errors++;
        // Weight tiles are laid out as B[k][output_channel]. This one-channel
        // trace therefore pads columns 1..3 while advancing k by tile row.
        if(weights_captured[0]!=8'h40||weights_captured[1]!=0||weights_captured[2]!=0||weights_captured[3]!=0||weights_captured[16]!=8'h41||weights_captured[17]!=0||weights_captured[18]!=0||weights_captured[19]!=0)errors++;
        if(error||errors)$fatal(1,"conv unfold trace failed %0d",errors);
        $display("Conv2D virtual im2col tile golden trace passed: activation and weight bursts verified");$finish;end
endmodule
