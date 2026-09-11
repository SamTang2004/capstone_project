module transformer_elementwise_ddr (
    input logic clk, rst_n, pipeline_enable, start, residual_mode,
    input logic [31:0] input0_base_addr, input1_base_addr, output_base_addr, total_bytes,
    input logic [3:0] lhs_right_shift, rhs_right_shift,
    output logic busy, done, error, op_start, op_done, memory_stall, input_stall, output_stall,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    output logic [6:0] read_req_len, output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input logic read_qword_valid, output logic read_qword_ready, input logic [63:0] read_qword_data,
    input logic [7:0] read_qword_keep, input logic read_qword_last,
    output logic write_req_valid, input logic write_req_ready, output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data, output logic [3:0] write_req_strb, output logic write_req_last,
    input logic memory_busy, memory_error
);
    typedef enum logic [3:0] {IDLE, PREP_READ, READ0_REQ, READ0_RECV, READ1_REQ, READ1_RECV,
        COMPUTE, WRITE_LOW, WRITE_HIGH, WRITE_ARM, WRITE_WAIT, FINISH, FAIL} state_t;
    state_t state;
    logic mode_r; logic [31:0] total_r;
    (* extract_enable = "no" *) logic [31:0] offset;
    logic [3:0] lhs_shift_r, rhs_shift_r;
    logic [63:0] lhs_r, rhs_r, gelu_result, residual_result, result_r;
    logic [7:0] keep_r; logic [6:0] chunk_bytes_r;
    logic gelu_in_ready,gelu_out_valid,residual_in_ready,residual_out_valid;

    gelu_int8 gelu(.input_valid(1'b1),.input_ready(gelu_in_ready),.input_vector(lhs_r),
        .output_valid(gelu_out_valid),.output_ready(1'b1),.output_vector(gelu_result));
    residual_add_int8 residual(.input_valid(1'b1),.input_ready(residual_in_ready),.lhs(lhs_r),.rhs(rhs_r),
        .lhs_right_shift(lhs_shift_r),.rhs_right_shift(rhs_shift_r),.output_valid(residual_out_valid),
        .output_ready(1'b1),.result(residual_result));

    assign busy=state!=IDLE&&state!=FINISH&&state!=FAIL; assign op_start=state==IDLE&&start; assign op_done=state==FINISH;
    assign read_req_valid=pipeline_enable&&(state==READ0_REQ||state==READ1_REQ);
    assign read_req_addr=(state==READ1_REQ?input1_base_addr:input0_base_addr)+offset;
    assign read_req_len=chunk_bytes_r; assign read_req_wide=0;assign read_req_qword=1;assign read_req_allow_ahead=0;
    assign read_qword_ready=pipeline_enable&&(state==READ0_RECV||state==READ1_RECV);
    assign write_req_valid=pipeline_enable&&(state==WRITE_LOW||state==WRITE_HIGH);
    assign write_req_addr=output_base_addr+offset+(state==WRITE_HIGH?4:0);
    assign write_req_data=state==WRITE_LOW?result_r[31:0]:result_r[63:32];
    assign write_req_strb=state==WRITE_LOW?keep_r[3:0]:keep_r[7:4];
    assign write_req_last=state==WRITE_HIGH||(state==WRITE_LOW&&keep_r[7:4]==0);
    assign memory_stall=busy&&((read_req_valid&&!read_req_ready)||(read_qword_ready&&!read_qword_valid)||(write_req_valid&&!write_req_ready));
    assign input_stall=(state==READ0_RECV||state==READ1_RECV)&&!read_qword_valid;
    assign output_stall=(state==WRITE_LOW||state==WRITE_HIGH)&&!write_req_ready;

    always_ff @(posedge clk) begin
        if(!rst_n)begin state<=IDLE;done<=0;error<=0;mode_r<=0;total_r<=0;offset<=0;lhs_shift_r<=0;rhs_shift_r<=0;lhs_r<=0;rhs_r<=0;keep_r<=0;chunk_bytes_r<=0;result_r<=0;end
        else begin done<=0;
            if(memory_error)begin error<=1;state<=FAIL;end
            else if(pipeline_enable)unique case(state)
                IDLE:if(start)begin error<=0;if(total_bytes==0)begin error<=1;state<=FAIL;end else begin mode_r<=residual_mode;total_r<=total_bytes;offset<=0;lhs_shift_r<=lhs_right_shift;rhs_shift_r<=rhs_right_shift;state<=PREP_READ;end end
                PREP_READ:begin chunk_bytes_r<=total_r-offset>=8?7'd8:total_r-offset;state<=READ0_REQ;end
                READ0_REQ:if(read_req_valid&&read_req_ready)state<=READ0_RECV;
                READ0_RECV:if(read_qword_valid)begin if(!read_qword_last)begin error<=1;state<=FAIL;end else begin lhs_r<=read_qword_data;keep_r<=read_qword_keep;state<=mode_r?READ1_REQ:COMPUTE;end end
                READ1_REQ:if(read_req_valid&&read_req_ready)state<=READ1_RECV;
                READ1_RECV:if(read_qword_valid)begin if(!read_qword_last||read_qword_keep!=keep_r)begin error<=1;state<=FAIL;end else begin rhs_r<=read_qword_data;state<=COMPUTE;end end
                COMPUTE:begin result_r<=mode_r?residual_result:gelu_result;state<=WRITE_LOW;end
                WRITE_LOW:if(write_req_valid&&write_req_ready)begin if(keep_r[7:4]!=0)state<=WRITE_HIGH;else if(offset+8>=total_r)state<=WRITE_ARM;else begin offset<=offset+8;state<=PREP_READ;end end
                WRITE_HIGH:if(write_req_valid&&write_req_ready)begin if(offset+8>=total_r)state<=WRITE_ARM;else begin offset<=offset+8;state<=PREP_READ;end end
                WRITE_ARM:state<=WRITE_WAIT;WRITE_WAIT:if(!memory_busy)state<=FINISH;
                FINISH:begin done<=1;state<=IDLE;end FAIL:begin done<=1;if(!start)state<=IDLE;end default:state<=IDLE;
            endcase
        end
    end
endmodule
