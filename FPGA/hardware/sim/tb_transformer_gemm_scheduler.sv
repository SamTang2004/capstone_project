module tb_transformer_gemm_scheduler;
    logic clk=0, rst_n=0, pipeline_enable=1, start=0, weight_int4=0;
    logic [31:0] a_base=32'h100,b_base=32'h400,c_base=32'h800,c_stride=8;
    logic [15:0] dim_m=3,dim_n=2,dim_k=2;
    logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
    logic [31:0] debug_state;
    logic read_req_valid,read_req_ready,read_req_wide,read_req_qword,read_req_allow_ahead;
    logic [31:0] read_req_addr; logic [6:0] read_req_len;
    logic read_qword_valid,read_qword_ready,read_qword_last;
    logic [63:0] read_qword_data; logic [7:0] read_qword_keep;
    logic write_req_valid,write_req_ready=1,write_req_last,memory_busy=0,memory_error=0;
    logic [31:0] write_req_addr,write_req_data; logic [3:0] write_req_strb;
    logic [7:0] mem[0:4095];
    logic pending; logic [31:0] pending_addr; logic [6:0] pending_len;
    integer errors=0;
    always #5 clk=~clk;

    transformer_gemm_scheduler #(.MAX_K(16)) dut(.*);
    assign read_req_ready = !pending && !read_qword_valid;

    always_ff @(posedge clk) begin
        if(!rst_n) begin pending<=0;read_qword_valid<=0;read_qword_data<=0;read_qword_keep<=0;read_qword_last<=0; end
        else begin
            if(read_req_valid&&read_req_ready) begin pending<=1;pending_addr<=read_req_addr;pending_len<=read_req_len; end
            if(pending&&!read_qword_valid) begin
                read_qword_data<=0;read_qword_keep<=0;
                for(int i=0;i<8;i++) if(i<pending_len) begin read_qword_data[i*8+:8]<=mem[pending_addr+i];read_qword_keep[i]<=1;end
                read_qword_last<=1;read_qword_valid<=1;pending<=0;
            end else if(read_qword_valid&&read_qword_ready) begin read_qword_valid<=0;read_qword_last<=0; end
            if(write_req_valid&&write_req_ready)
                for(int i=0;i<4;i++) if(write_req_strb[i]) mem[write_req_addr+i]<=write_req_data[i*8+:8];
        end
    end

    task automatic put_qword(input int addr,input logic [63:0] value);
        for(int i=0;i<8;i++)mem[addr+i]=value[i*8+:8];
    endtask
    task automatic expect_i32(input int addr,input integer expected);
        logic signed [31:0] got;
        begin got={mem[addr+3],mem[addr+2],mem[addr+1],mem[addr]};if(got!==expected)begin $error("addr=%h expected=%0d got=%0d",addr,expected,got);errors++;end end
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

        for(int i='h800;i<'h818;i++)mem[i]=0;
        weight_int4=1;
        // W4 vectors: k0=[-2,3], k1=[4,-1].
        mem['h400]=8'h3e;mem['h401]=0;mem['h402]=0;mem['h403]=0;
        mem['h404]=8'hf4;mem['h405]=0;mem['h406]=0;mem['h407]=0;
        launch();
        expect_i32('h800,6);expect_i32('h804,1);expect_i32('h808,10);
        expect_i32('h80c,5);expect_i32('h810,14);expect_i32('h814,9);
        if(error||errors)$fatal(1,"Transformer scheduler failed errors=%0d",errors);
        $display("Transformer tiled GEMM scheduler golden trace passed");$finish;
    end
    initial begin #200000;$fatal(1,"Scheduler timeout state=%0d",dut.state);end
endmodule
