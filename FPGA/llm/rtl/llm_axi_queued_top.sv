// Compatibility-first integration: reuse the existing AXI slave and replay
// queued register snapshots into the unchanged DeiT DDR/GEMM top.
// Replay latency is launch overhead, NOT array T0.
module llm_axi_queued_top (
    input logic aclk, aresetn,
    input logic [7:0] s_axi_awaddr, input logic s_axi_awvalid, output logic s_axi_awready,
    input logic [31:0] s_axi_wdata, input logic [3:0] s_axi_wstrb, input logic s_axi_wvalid, output logic s_axi_wready,
    output logic [1:0] s_axi_bresp, output logic s_axi_bvalid, input logic s_axi_bready,
    input logic [7:0] s_axi_araddr, input logic s_axi_arvalid, output logic s_axi_arready,
    output logic [31:0] s_axi_rdata, output logic [1:0] s_axi_rresp, output logic s_axi_rvalid, input logic s_axi_rready,
    output logic mem_cmd_valid, input logic mem_cmd_ready, output logic mem_cmd_write,
    output logic [31:0] mem_cmd_addr, output logic [3:0] mem_cmd_len,
    output logic [31:0] mem_cmd_wdata, output logic [3:0] mem_cmd_wstrb,
    output logic mem_write_valid, input logic mem_write_ready,
    output logic [31:0] mem_write_data, output logic [3:0] mem_write_strb, output logic mem_write_last,
    input logic mem_read_valid, output logic mem_read_ready, input logic [63:0] mem_read_data, input logic mem_read_last,
    input logic [3:0] mem_debug_arlen, input logic [2:0] mem_debug_arsize,
    input logic [4:0] mem_debug_rbeats, input logic [1:0] mem_debug_rresp
);
    logic [511:0] staging, queued, active, perf;
    logic [31:0] ctrl, debug_ctrl, status_word;
    logic doorbell_q, ack_q, reject, faulted;
    logic q_ready, q_valid, q_pop, enqueue;
    logic [2:0] occupancy;
    logic completion_valid;
    logic [31:0] result_batch, result_task, result_status, cycles, result_cycles;
    typedef enum logic [3:0] {IDLE, SEND, RESPONSE, POLL_ADDR, POLL_DATA, CLEAR_SEND, CLEAR_RESP, COMPLETE} state_t;
    state_t state;
    logic [3:0] register_index;
    logic aw_sent, w_sent;
    logic [7:0] awaddr, araddr;
    logic [31:0] wdata, rdata;
    logic awvalid, awready, wvalid, wready, bvalid, bready;
    logic arvalid, arready, rvalid, rready;
    logic [1:0] bresp, rresp;
    wire acknowledge = debug_ctrl[0] != ack_q;
    assign enqueue = ctrl[0] && !doorbell_q;
    assign q_pop = state == IDLE && !faulted;
    assign status_word = {23'd0, faulted, occupancy, reject, completion_valid, q_ready, (state != IDLE), 1'b0};
    always_comb begin
        perf = '0;
        perf[0*32+:32] = 32'h4c4c4d01; // LLM ABI v1
        perf[1*32+:32] = result_batch;
        perf[2*32+:32] = result_task;
        perf[3*32+:32] = result_status;
        perf[4*32+:32] = result_cycles;
    end
    axi_lite_control host_control (
        .aclk,.aresetn, .s_axi_awaddr,.s_axi_awvalid,.s_axi_awready,
        .s_axi_wdata,.s_axi_wstrb,.s_axi_wvalid,.s_axi_wready,
        .s_axi_bresp,.s_axi_bvalid,.s_axi_bready,
        .s_axi_araddr,.s_axi_arvalid,.s_axi_arready,
        .s_axi_rdata,.s_axi_rresp,.s_axi_rvalid,.s_axi_rready,
        .status_word, .perf_regs_flat(perf), .debug_regs_flat(576'd0),
        .extended_perf_regs_flat(192'd0), .ctrl_word(ctrl),
        .debug_ctrl_word(debug_ctrl), .regs_flat(staging)
    );
    llm_command_queue #(.WIDTH(512),.DEPTH(4)) queue (
        .clk(aclk),.rst_n(aresetn),.push_valid(enqueue),.push_ready(q_ready),
        .push_data(staging),.pop_valid(q_valid),.pop_ready(q_pop),.pop_data(queued),.occupancy
    );
    assign awaddr = state == CLEAR_SEND ? 8'd0 : {2'd0,register_index,2'd0};
    assign wdata = state == CLEAR_SEND ? 32'd0 :
                   register_index == 11 ? 32'd8 :
                   register_index == 12 ? 32'd1 : active[register_index*32+:32];
    assign awvalid = (state == SEND || state == CLEAR_SEND) && !aw_sent;
    assign wvalid = (state == SEND || state == CLEAR_SEND) && !w_sent;
    assign bready = state == RESPONSE || state == CLEAR_RESP;
    assign araddr = 8'h04;
    assign arvalid = state == POLL_ADDR;
    assign rready = state == POLL_DATA;
    bcuda_transformer_ddr_top backend (
        .aclk,.aresetn,.s_axi_awaddr(awaddr),.s_axi_awvalid(awvalid),.s_axi_awready(awready),
        .s_axi_wdata(wdata),.s_axi_wstrb(4'hf),.s_axi_wvalid(wvalid),.s_axi_wready(wready),
        .s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_araddr(araddr),.s_axi_arvalid(arvalid),.s_axi_arready(arready),
        .s_axi_rdata(rdata),.s_axi_rresp(rresp),.s_axi_rvalid(rvalid),.s_axi_rready(rready),
        .mem_cmd_valid,.mem_cmd_ready,.mem_cmd_write,.mem_cmd_addr,.mem_cmd_len,.mem_cmd_wdata,.mem_cmd_wstrb,
        .mem_write_valid,.mem_write_ready,.mem_write_data,.mem_write_strb,.mem_write_last,
        .mem_read_valid,.mem_read_ready,.mem_read_data,.mem_read_last,
        .mem_debug_arlen,.mem_debug_arsize,.mem_debug_rbeats,.mem_debug_rresp
    );
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state<=IDLE; active<=0; register_index<=2; aw_sent<=0; w_sent<=0;
            doorbell_q<=0; ack_q<=0; reject<=0; faulted<=0; completion_valid<=0;
            result_batch<=0; result_task<=0; result_status<=0; cycles<=0; result_cycles<=0;
        end else begin
            doorbell_q<=ctrl[0]; ack_q<=debug_ctrl[0];
            if(acknowledge) reject<=0;
            if(enqueue && !q_ready) reject<=1;
            if(state!=IDLE && state!=COMPLETE) cycles<=cycles+1'b1;
            case(state)
                IDLE: if(q_valid && !faulted) begin
                    active<=queued; register_index<=2; aw_sent<=0; w_sent<=0;
                    result_batch<=queued[11*32+:32]; result_task<=queued[12*32+:32];
                    result_status<=0; cycles<=0;
                    // v1 accepts plain GEMM only (START and optional W4).
                    if ((queued[31:0] & ~32'h00000801)!=0) begin
                        result_status<=32'h80000001; result_cycles<=0;
                        completion_valid<=1; state<=COMPLETE;
                    end else state<=SEND;
                end
                SEND,CLEAR_SEND: begin
                    if(awvalid && awready) aw_sent<=1;
                    if(wvalid && wready) w_sent<=1;
                    if((aw_sent || awready) && (w_sent || wready))
                        state<=state==SEND ? RESPONSE : CLEAR_RESP;
                end
                RESPONSE: if(bvalid) begin
                    aw_sent<=0; w_sent<=0;
                    if(bresp!=0) begin faulted<=1; result_status<=32'h80000002; state<=CLEAR_SEND; end
                    else if(register_index==0) state<=POLL_ADDR;
                    else begin register_index<=register_index==15 ? 0 : register_index+1'b1; state<=SEND; end
                end
                POLL_ADDR: if(arready) state<=POLL_DATA;
                POLL_DATA: if(rvalid) begin
                    if(rresp!=0 || rdata[2] || (rdata[1] && !rdata[0])) begin
                        result_status<=rresp!=0 ? 32'h80000002 : rdata;
                        if(rresp!=0 || rdata[2]) faulted<=1;
                        aw_sent<=0; w_sent<=0; state<=CLEAR_SEND;
                    end else state<=POLL_ADDR;
                end
                CLEAR_RESP: if(bvalid) begin
                    if(bresp!=0) begin faulted<=1; result_status<=32'h80000002; end
                    result_cycles<=cycles+1'b1; completion_valid<=1; state<=COMPLETE;
                end
                COMPLETE: if(acknowledge) begin completion_valid<=0; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
