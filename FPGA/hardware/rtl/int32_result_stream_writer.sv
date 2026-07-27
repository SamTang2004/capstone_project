`timescale 1ns/1ps
module int32_result_stream_writer (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    output logic               input_ready,
    input  logic               input_array,
    input  logic [5:0]         input_addr,
    input  logic signed [31:0] input_data,
    input  logic               input_last,
    input  logic               conv_mode,
    input  logic [31:0]        output_base_addr,
    input  logic [31:0]        row_stride_bytes,
    input  logic [15:0]        m0,
    input  logic [15:0]        n0,
    input  logic [15:0]        out_h,
    input  logic [15:0]        out_w,
    output logic               write_req_valid,
    input  logic               write_req_ready,
    output logic [31:0]        write_req_addr,
    output logic [31:0]        write_req_data,
    output logic [3:0]         write_req_strb,
    output logic               write_req_last
);

    typedef enum logic [2:0] { IDLE, INDEX0, INDEX1, ADDRESS, WRITE } state_t;
    state_t state;
    logic conv_r;
    logic [31:0] base_r, stride_r;
    logic [15:0] global_row_r, global_col_r;
    logic [15:0] out_h_r, out_w_r;
    logic [31:0] index_stage0_r;
    logic [31:0] index_stage1_r;
    logic signed [31:0] data_r;
    logic last_r;

    assign input_ready = state == IDLE;
    assign write_req_valid = state == WRITE;
    assign write_req_data = data_r;
    assign write_req_strb = 4'hf;
    assign write_req_last = last_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            conv_r <= 1'b0;
            base_r <= '0;
            stride_r <= '0;
            global_row_r <= '0;
            global_col_r <= '0;
            out_h_r <= '0;
            out_w_r <= '0;
            index_stage0_r <= '0;
            index_stage1_r <= '0;
            data_r <= '0;
            last_r <= 1'b0;
            write_req_addr <= '0;
        end else begin
            unique case (state)
                IDLE: if (input_valid) begin
                    conv_r <= conv_mode;
                    base_r <= output_base_addr;
                    stride_r <= row_stride_bytes;
                    global_row_r <= m0 + input_addr[5:3];
                    global_col_r <= n0 + {input_array, 3'b000} + input_addr[2:0];
                    out_h_r <= out_h;
                    out_w_r <= out_w;
                    data_r <= input_data;
                    last_r <= input_last;
                    state <= INDEX0;
                end

                INDEX0: begin
                    index_stage0_r <= conv_r ?
                                      global_col_r * out_h_r :
                                      global_row_r * stride_r;
                    state <= INDEX1;
                end

                INDEX1: begin
                    index_stage1_r <= conv_r ?
                                      index_stage0_r * out_w_r :
                                      index_stage0_r + (global_col_r << 2);
                    state <= ADDRESS;
                end

                ADDRESS: begin
                    write_req_addr <= base_r +
                                      (conv_r ?
                                       ((index_stage1_r + global_row_r) << 2) :
                                       index_stage1_r);
                    state <= WRITE;
                end

                WRITE: if (write_req_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
