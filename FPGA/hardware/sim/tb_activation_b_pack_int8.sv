module tb_activation_b_pack_int8;
    logic clk=0,rst_n=0,pipeline_enable=1,start=0,transpose=0,source_row_major=0; logic[15:0] reduction=3,outputs=2; always #5 clk=~clk;
    logic busy,done,error,op_start,op_done,memory_stall,input_stall,output_stall;
    logic read_req_valid,read_req_ready=1,read_req_wide,read_req_qword,read_req_allow_ahead;
    logic[31:0]read_req_addr;logic[6:0]read_req_len;logic read_data_valid=0,read_data_ready,read_data_last=0;logic[7:0]read_data;
    logic write_req_valid,write_req_ready=1,write_req_last;logic[31:0]write_req_addr,write_req_data;logic[3:0]write_req_strb;
    logic memory_busy=0,memory_error=0;logic[7:0]mem[0:2047];integer errors=0;

    activation_b_pack_int8 dut(.clk,.rst_n,.pipeline_enable,.start,.transpose,.source_row_major,
        .input_base_addr(32'h100),.output_base_addr(32'h500),.reduction,.outputs,
        .busy,.done,.error,.op_start,.op_done,.memory_stall,.input_stall,.output_stall,
        .read_req_valid,.read_req_ready,.read_req_addr,.read_req_len,.read_req_wide,.read_req_qword,.read_req_allow_ahead,
        .read_data_valid,.read_data_ready,.read_data,.read_data_last,
        .write_req_valid,.write_req_ready,.write_req_addr,.write_req_data,.write_req_strb,.write_req_last,.memory_busy,.memory_error);

    always_ff @(posedge clk) begin
        read_data_valid<=0;read_data_last<=0;
        if(read_req_valid&&read_req_ready)begin read_data<=mem[read_req_addr];read_data_valid<=1;read_data_last<=1;end
        if(write_req_valid&&write_req_ready)
            for(int lane=0;lane<4;lane++)if(write_req_strb[lane])mem[write_req_addr+lane]<=write_req_data[lane*8+:8];
    end

    task automatic clear_output;
        begin
            for(int i='h500;i<'h540;i++)mem[i]=0;
        end
    endtask
    task automatic launch(input logic t);
        begin
            transpose=t;@(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(posedge clk);
            if(error) begin $error("unexpected error");errors++;end
        end
    endtask

    initial begin
        for(int i=0;i<2048;i++)mem[i]=0;
        // Tiled activation for row-major [[1,2],[3,4],[5,6]].
        mem['h100]=1;mem['h101]=3;mem['h102]=5;
        mem['h108]=2;mem['h109]=4;mem['h10a]=6;
        repeat(4)@(posedge clk);rst_n=1;
        // Normal B is 3x2: vectors [1,2], [3,4], [5,6].
        launch(0);
        if({mem['h501],mem['h500]}!==16'h0201||{mem['h509],mem['h508]}!==16'h0403||{mem['h511],mem['h510]}!==16'h0605)errors++;
        for(int base='h500;base<'h518;base+=8)for(int lane=2;lane<8;lane++)if(mem[base+lane]!==0)errors++;
        // Transposed B is 2x3: vectors [1,3,5], [2,4,6].
        clear_output();reduction=2;outputs=3;launch(1);
        if({mem['h502],mem['h501],mem['h500]}!==24'h050301||{mem['h50a],mem['h509],mem['h508]}!==24'h060402)errors++;
        // Row-major source uses the same transpose mapping to produce tiled A.
        clear_output();source_row_major=1;
        mem['h100]=1;mem['h101]=2;mem['h102]=3;mem['h103]=4;mem['h104]=5;mem['h105]=6;
        launch(1);
        if({mem['h502],mem['h501],mem['h500]}!==24'h050301||{mem['h50a],mem['h509],mem['h508]}!==24'h060402)errors++;
        if(errors)$fatal(1,"Activation B pack failed errors=%0d",errors);
        $display("Dynamic activation transpose/B-pack golden trace passed");$finish;
    end
    initial begin#100000;$fatal(1,"timeout state=%0d",dut.state);end
endmodule
