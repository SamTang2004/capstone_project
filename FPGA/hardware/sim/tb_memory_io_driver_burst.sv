module tb_memory_io_driver_burst;
    logic clk=0,rst_n=0,read_req_valid,read_req_ready,read_data_valid,read_data_ready;
    logic [31:0] read_req_addr; logic [4:0] read_req_len; logic [7:0] read_data; logic read_data_last;
    logic write_req_valid,write_req_ready,write_req_last; logic [31:0] write_req_addr,write_req_data;
    logic [3:0] write_req_strb,mem_cmd_len,mem_cmd_wstrb,mem_write_strb;
    logic mem_cmd_valid,mem_cmd_ready=1,mem_cmd_write,mem_write_valid,mem_write_ready=1,mem_write_last;
    logic [31:0] mem_cmd_addr,mem_cmd_wdata,mem_write_data,mem_read_data;
    logic mem_read_valid=0,mem_read_ready,mem_read_last=0,busy,error;
    integer beat,reads,errors=0; logic [31:0] mem[0:63]; logic [31:0] active_base; logic [3:0] active_len;
    always #5 clk=~clk;
    initial begin repeat(5000)@(posedge clk);$fatal(1,"memory IO burst timeout busy=%0b read_pending=%0b fill=%0b deliver=%0b write_count=%0d flush=%0b send=%0b inflight=%0b",busy,dut.read_pending,dut.fill_active,dut.deliver_active,dut.write_count,dut.write_flush_pending,dut.write_send_active,dut.write_inflight);end
    memory_io_driver dut(.*,.clk(clk),.rst_n(rst_n));

    always_ff @(posedge clk) begin
        if(!rst_n) begin mem_read_valid<=0;mem_read_last<=0;beat<=0;reads<=0; end else begin
            if(mem_cmd_valid&&mem_cmd_ready&&!mem_cmd_write) begin
                beat<=0;active_base<=mem_cmd_addr;active_len<=mem_cmd_len;mem_read_data<=mem[mem_cmd_addr>>2];mem_read_valid<=1;mem_read_last<=(mem_cmd_len==0);reads<=reads+1;
            end else if(mem_read_valid&&mem_read_ready) begin
                if(mem_read_last) begin mem_read_valid<=0;mem_read_last<=0; end
                else begin beat<=beat+1;mem_read_data<=mem[(active_base>>2)+beat+1];mem_read_last<=(beat+1==active_len); end
            end
            if(mem_write_valid&&mem_write_ready) mem[(active_base>>2)+beat]<=mem_write_data;
            if(mem_cmd_valid&&mem_cmd_ready&&mem_cmd_write) begin beat<=0;active_base<=mem_cmd_addr;active_len<=mem_cmd_len;end
            else if(mem_write_valid&&mem_write_ready&&!mem_write_last) beat<=beat+1;
        end
    end

    task automatic get_byte(input [31:0] addr,input [7:0] expected);
        begin @(negedge clk);read_req_addr=addr;read_req_len=1;read_req_valid=1;wait(read_req_ready);@(negedge clk);read_req_valid=0;wait(read_data_valid);#1;if(read_data!==expected||!read_data_last)errors++;@(negedge clk);end
    endtask
    task automatic get_run(input [31:0] addr,input integer length);
        begin
            @(negedge clk);read_req_addr=addr;read_req_len=length;read_req_valid=1;wait(read_req_ready);@(negedge clk);read_req_valid=0;
            for(int i=0;i<length;i++) begin
                wait(read_data_valid);#1;
                if(read_data!==(addr+i) || read_data_last!==(i==length-1)) errors++;
                @(posedge clk);
            end
        end
    endtask
    task automatic put_single_word(input [31:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            write_req_addr = addr;
            write_req_data = data;
            write_req_strb = 4'hf;
            write_req_last = 1'b1;
            write_req_valid = 1'b1;
            // Hold valid through a full clock edge after ready is observed.
            // A ready transition at the sampling edge is not a handshake.
            while (!write_req_ready) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            write_req_valid = 1'b0;
            write_req_last = 1'b0;
        end
    endtask
    initial begin
        for(int i=0;i<64;i++)mem[i]=32'h03020100+(i*32'h04040404);
        read_req_valid=0;read_req_addr=0;read_req_len=1;read_data_ready=1;write_req_valid=0;write_req_addr=0;
        write_req_data=0;write_req_strb=4'hf;write_req_last=0;
        repeat(3)@(posedge clk);rst_n=1;
        get_byte(32'h04,8'h04);get_byte(32'h07,8'h07);get_run(32'h30,14);
        if(reads!=1)errors++;
        get_run(32'h44,16);
        if(reads!=2)errors++;
        for(int i=0;i<4;i++) begin @(negedge clk);write_req_valid=1;write_req_addr=32'h80+i*4;write_req_data=32'habc00000+i;write_req_last=(i==3);wait(write_req_ready);end
        @(negedge clk);write_req_valid=0;write_req_last=0;wait(!busy);
        for(int i=0;i<4;i++)if(mem[32+i]!==32'habc00000+i)errors++;
        // Back-to-back short writes exercise the handoff from an AXI write
        // response to the next coalesced burst. This is the board path used
        // by requantize and activation tails.
        for(int i=0;i<5;i++) put_single_word(32'hA0+i*4, 32'hd00d0000+i);
        wait(!busy);
        for(int i=0;i<5;i++) begin
            if(mem[40+i]!==32'hd00d0000+i) begin
                $display("short burst mismatch index=%0d expected=0x%08h got=0x%08h", i, 32'hd00d0000+i, mem[40+i]);
                errors++;
            end
        end
        if(error||errors)$fatal(1,"memory IO burst test failed errors=%0d",errors);
        $display("Memory IO burst golden trace passed: cached read and 4-beat coalesced write");$finish;
    end
endmodule
