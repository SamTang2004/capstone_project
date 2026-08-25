`timescale 1ns/1ps
module heatvit_token_pruner_ddr #(
    parameter int MAX_TOKENS = 256,
    parameter int MAX_FEATURES = 192
) (
    input  logic clk, rst_n, pipeline_enable, start, weight_int4,
    input  logic [31:0] input_base_addr, weight_base_addr, output_base_addr,
    input  logic [15:0] input_tokens, output_tokens, features,
    input  logic [31:0] package_reciprocal_q16,
    output logic busy, done, error, op_start, op_done,
    output logic memory_stall, input_stall, output_stall,
    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [6:0] read_req_len,
    output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input  logic read_data_valid, output logic read_data_ready,
    input  logic [7:0] read_data, input logic read_data_last,
    input  logic read_qword_valid, output logic read_qword_ready,
    input  logic [63:0] read_qword_data, input logic [7:0] read_qword_keep,
    input  logic read_qword_last,
    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last,
    input logic memory_busy, memory_error
);
    localparam int MAX_CHUNKS = (MAX_FEATURES + 7) / 8;
    typedef enum logic [5:0] {
        IDLE, WEIGHT_REQ, WEIGHT_RECV, ROW_START, TOKEN_REQ, TOKEN_RECV,
        SCORE_LOW, SCORE_HIGH, SCORE_WAIT, PICK_INIT, PICK_SCAN, PICK_COMMIT,
        SUM_CLEAR, SCAN_ROW, SCAN_REQ, SCAN_RECV, KEEP_WRITE_LOW, KEEP_WRITE_HIGH,
        PACKAGE_ACCUM, PACKAGE_READ, PACKAGE_PREP, PACKAGE_SCALE, PACKAGE_WRITE, WRITE_DRAIN, FINISH, FAIL
    } state_t;
    state_t state;

    (* ram_style = "distributed" *) logic signed [7:0] weight_mem [0:MAX_FEATURES-1];
    (* ram_style = "block" *) logic signed [39:0] score_mem [0:MAX_TOKENS-1];
    /* Eight narrow banks turn the eight-byte accumulation into one write per
     * bank.  A feature-major array caused Vivado to dissolve the complete
     * package cache into a 192-way distributed-register mux. */
    (* ram_style = "distributed" *) logic signed [31:0] package_sum0 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum1 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum2 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum3 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum4 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum5 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum6 [0:MAX_CHUNKS-1];
    (* ram_style = "distributed" *) logic signed [31:0] package_sum7 [0:MAX_CHUNKS-1];
    logic selected [0:MAX_TOKENS-1];

    logic [15:0] tokens_r, outputs_r, features_r, chunks_r;
    logic [31:0] reciprocal_r;
    logic weight_int4_r;
    logic [15:0] feature_index, row_index, chunk_index;
    logic [15:0] pick_index, scan_index, best_index;
    logic best_valid;
    logic signed [39:0] best_score;
    logic [63:0] token_qword;
    logic [31:0] package_word;
    logic signed [63:0] package_product;
    logic signed [31:0] package_sum_value;
    logic signed [31:0] package_sum_r;
    logic [31:0] weight_addr_cursor, input_addr_cursor, output_addr_cursor;
    integer score_lane, clear_lane;

    logic scorer_start, scorer_valid, scorer_last, scorer_score_valid;
    logic signed [7:0] scorer_activation [0:3], scorer_weight [0:3];
    logic signed [39:0] scorer_score;
    heatvit_token_scorer_4lane scorer (
        .clk, .rst_n, .start_token(scorer_start), .feature_valid(scorer_valid),
        .feature_last(scorer_last), .activation(scorer_activation), .weight(scorer_weight),
        .bias(32'sd0), .feature_ready(), .score_valid(scorer_score_valid), .score(scorer_score)
    );

    function automatic [7:0] package_value(input logic signed [63:0] product);
        logic signed [31:0] rounded;
        begin
            rounded = product >= 0 ? (product + 64'sd32768) >>> 16
                                   : -(((-product) + 64'sd32768) >>> 16);
            if (rounded > 127) package_value = 8'h7f;
            else if (rounded < -128) package_value = 8'h80;
            else package_value = rounded[7:0];
        end
    endfunction

    wire row_kept = row_index == 0 || selected[row_index];
    always_comb begin
        scorer_start = state == ROW_START && row_index != 0;
        scorer_valid = state == SCORE_LOW || state == SCORE_HIGH;
        scorer_last = state == SCORE_HIGH && chunk_index + 1 == chunks_r;
        for (score_lane = 0; score_lane < 4; score_lane = score_lane + 1) begin
            scorer_activation[score_lane] = state == SCORE_HIGH
                ? $signed(token_qword[(score_lane+4)*8 +: 8]) : $signed(token_qword[score_lane*8 +: 8]);
            scorer_weight[score_lane] = state == SCORE_HIGH
                ? weight_mem[chunk_index*8 + score_lane + 4] : weight_mem[chunk_index*8 + score_lane];
        end
    end

    always_comb begin
        unique case (feature_index[2:0])
            3'd0: package_sum_value = package_sum0[feature_index >> 3];
            3'd1: package_sum_value = package_sum1[feature_index >> 3];
            3'd2: package_sum_value = package_sum2[feature_index >> 3];
            3'd3: package_sum_value = package_sum3[feature_index >> 3];
            3'd4: package_sum_value = package_sum4[feature_index >> 3];
            3'd5: package_sum_value = package_sum5[feature_index >> 3];
            3'd6: package_sum_value = package_sum6[feature_index >> 3];
            default: package_sum_value = package_sum7[feature_index >> 3];
        endcase
    end

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign done = state == FINISH || state == FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;
    assign read_req_valid = pipeline_enable && (state == WEIGHT_REQ || state == TOKEN_REQ || state == SCAN_REQ);
    assign read_req_addr = state == WEIGHT_REQ
        ? weight_addr_cursor : input_addr_cursor;
    assign read_req_len = state == WEIGHT_REQ ? 7'd1 : 7'd8;
    assign read_req_wide = state == TOKEN_REQ || state == SCAN_REQ;
    assign read_req_qword = state == TOKEN_REQ || state == SCAN_REQ;
    assign read_req_allow_ahead = 1'b0;
    assign read_data_ready = pipeline_enable && state == WEIGHT_RECV;
    assign read_qword_ready = pipeline_enable && (state == TOKEN_RECV || state == SCAN_RECV);
    assign write_req_valid = pipeline_enable &&
        (state == KEEP_WRITE_LOW || state == KEEP_WRITE_HIGH || state == PACKAGE_WRITE);
    assign write_req_addr = output_addr_cursor + (state == KEEP_WRITE_HIGH ? 4 : 0);
    assign write_req_data = state == PACKAGE_WRITE ? package_word
        : (state == KEEP_WRITE_HIGH ? token_qword[63:32] : token_qword[31:0]);
    assign write_req_strb = 4'hf;
    assign write_req_last = state == KEEP_WRITE_HIGH || state == PACKAGE_WRITE;
    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
        (read_data_ready && !read_data_valid) || (read_qword_ready && !read_qword_valid) ||
        (write_req_valid && !write_req_ready));
    assign input_stall = (state == WEIGHT_RECV && !read_data_valid) ||
                         ((state == TOKEN_RECV || state == SCAN_RECV) && !read_qword_valid);
    assign output_stall = write_req_valid && !write_req_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; error <= 0; tokens_r <= 0; outputs_r <= 0; features_r <= 0;
            chunks_r <= 0; reciprocal_r <= 0; weight_int4_r <= 0; feature_index <= 0;
            row_index <= 0; chunk_index <= 0; pick_index <= 0;
            scan_index <= 0; best_index <= 0; best_valid <= 0;
            best_score <= 0; token_qword <= 0; package_word <= 0; package_product <= 0; package_sum_r <= 0;
            weight_addr_cursor <= 0; input_addr_cursor <= 0; output_addr_cursor <= 0;
        end else if (memory_error) begin
            error <= 1; state <= FAIL;
        end else if (pipeline_enable) begin
            unique case (state)
                IDLE: if (start) begin
                    error <= 0;
                    if (input_tokens < 3 || output_tokens < 2 || output_tokens >= input_tokens ||
                        input_tokens > MAX_TOKENS || features == 0 || features > MAX_FEATURES ||
                        features[2:0] != 0 || package_reciprocal_q16 == 0) begin
                        error <= 1; state <= FAIL;
                    end else begin
                        tokens_r <= input_tokens; outputs_r <= output_tokens; features_r <= features;
                        chunks_r <= features >> 3; reciprocal_r <= package_reciprocal_q16;
                        weight_int4_r <= weight_int4; feature_index <= 0;
                        weight_addr_cursor <= weight_base_addr; input_addr_cursor <= input_base_addr;
                        output_addr_cursor <= output_base_addr; state <= WEIGHT_REQ;
                    end
                end
                WEIGHT_REQ: if (read_req_valid && read_req_ready) state <= WEIGHT_RECV;
                WEIGHT_RECV: if (read_data_valid) begin
                    if (!read_data_last) begin error <= 1; state <= FAIL; end
                    else begin
                        weight_mem[feature_index] <= weight_int4_r
                            ? $signed({{4{read_data[3]}}, read_data[3:0]}) : $signed(read_data);
                        weight_addr_cursor <= weight_addr_cursor + (weight_int4_r ? 4 : 8);
                        if (feature_index + 1 == features_r) begin
                            row_index <= 0; chunk_index <= 0; state <= ROW_START;
                        end else begin feature_index <= feature_index + 1; state <= WEIGHT_REQ; end
                    end
                end
                ROW_START: begin chunk_index <= 0; state <= TOKEN_REQ; end
                TOKEN_REQ: if (read_req_valid && read_req_ready) state <= TOKEN_RECV;
                TOKEN_RECV: if (read_qword_valid) begin
                    if (!read_qword_last || read_qword_keep != 8'hff) begin error <= 1; state <= FAIL; end
                    else begin
                        token_qword <= read_qword_data;
                        input_addr_cursor <= input_addr_cursor + 8;
                        state <= row_index == 0 ? SCORE_HIGH : SCORE_LOW;
                    end
                end
                SCORE_LOW: state <= SCORE_HIGH;
                SCORE_HIGH: begin
                    if (row_index == 0) begin
                        if (chunk_index + 1 == chunks_r) begin row_index <= 1; state <= ROW_START; end
                        else begin chunk_index <= chunk_index + 1; state <= TOKEN_REQ; end
                    end else if (chunk_index + 1 == chunks_r) state <= SCORE_WAIT;
                    else begin chunk_index <= chunk_index + 1; state <= TOKEN_REQ; end
                end
                SCORE_WAIT: if (scorer_score_valid) begin
                    score_mem[row_index] <= scorer_score;
                    if (row_index + 1 == tokens_r) begin pick_index <= 0; state <= PICK_INIT; end
                    else begin row_index <= row_index + 1; state <= ROW_START; end
                end
                PICK_INIT: begin
                    if (pick_index == outputs_r - 2) begin chunk_index <= 0; state <= SUM_CLEAR; end
                    else begin scan_index <= 1; best_valid <= 0; best_index <= 0; state <= PICK_SCAN; end
                end
                PICK_SCAN: begin
                    if (scan_index == tokens_r) state <= PICK_COMMIT;
                    else begin
                        if (!selected[scan_index] && (!best_valid || score_mem[scan_index] > best_score)) begin
                            best_valid <= 1; best_score <= score_mem[scan_index]; best_index <= scan_index;
                        end
                        scan_index <= scan_index + 1;
                    end
                end
                PICK_COMMIT: begin
                    if (!best_valid) begin error <= 1; state <= FAIL; end
                    else begin selected[best_index] <= 1; pick_index <= pick_index + 1; state <= PICK_INIT; end
                end
                SUM_CLEAR: begin
                    package_sum0[chunk_index] <= 0; package_sum1[chunk_index] <= 0;
                    package_sum2[chunk_index] <= 0; package_sum3[chunk_index] <= 0;
                    package_sum4[chunk_index] <= 0; package_sum5[chunk_index] <= 0;
                    package_sum6[chunk_index] <= 0; package_sum7[chunk_index] <= 0;
                    if (chunk_index + 1 == chunks_r) begin
                        row_index <= 0; chunk_index <= 0;
                        input_addr_cursor <= input_base_addr; output_addr_cursor <= output_base_addr;
                        state <= SCAN_ROW;
                    end else chunk_index <= chunk_index + 1;
                end
                SCAN_ROW: begin
                    if (row_index == tokens_r) begin feature_index <= 0; state <= PACKAGE_READ; end
                    else begin chunk_index <= 0; state <= SCAN_REQ; end
                end
                SCAN_REQ: if (read_req_valid && read_req_ready) state <= SCAN_RECV;
                SCAN_RECV: if (read_qword_valid) begin
                    if (!read_qword_last || read_qword_keep != 8'hff) begin error <= 1; state <= FAIL; end
                    else begin
                        token_qword <= read_qword_data; input_addr_cursor <= input_addr_cursor + 8;
                        state <= row_kept ? KEEP_WRITE_LOW : PACKAGE_ACCUM;
                    end
                end
                KEEP_WRITE_LOW: if (write_req_valid && write_req_ready) state <= KEEP_WRITE_HIGH;
                KEEP_WRITE_HIGH: if (write_req_valid && write_req_ready) begin
                    output_addr_cursor <= output_addr_cursor + 8;
                    if (chunk_index + 1 == chunks_r) begin
                        row_index <= row_index + 1; state <= SCAN_ROW;
                    end else begin chunk_index <= chunk_index + 1; state <= SCAN_REQ; end
                end
                PACKAGE_ACCUM: begin
                    package_sum0[chunk_index] <= package_sum0[chunk_index] + $signed(token_qword[7:0]);
                    package_sum1[chunk_index] <= package_sum1[chunk_index] + $signed(token_qword[15:8]);
                    package_sum2[chunk_index] <= package_sum2[chunk_index] + $signed(token_qword[23:16]);
                    package_sum3[chunk_index] <= package_sum3[chunk_index] + $signed(token_qword[31:24]);
                    package_sum4[chunk_index] <= package_sum4[chunk_index] + $signed(token_qword[39:32]);
                    package_sum5[chunk_index] <= package_sum5[chunk_index] + $signed(token_qword[47:40]);
                    package_sum6[chunk_index] <= package_sum6[chunk_index] + $signed(token_qword[55:48]);
                    package_sum7[chunk_index] <= package_sum7[chunk_index] + $signed(token_qword[63:56]);
                    if (chunk_index + 1 == chunks_r) begin row_index <= row_index + 1; state <= SCAN_ROW; end
                    else begin chunk_index <= chunk_index + 1; state <= SCAN_REQ; end
                end
                PACKAGE_READ: begin package_sum_r <= package_sum_value; state <= PACKAGE_PREP; end
                PACKAGE_PREP: begin
                    package_product <= package_sum_r *
                                       $signed({1'b0, reciprocal_r[30:0]});
                    state <= PACKAGE_SCALE;
                end
                PACKAGE_SCALE: begin
                    package_word[feature_index[1:0]*8 +: 8] <= package_value(package_product);
                    if (feature_index[1:0] == 2'd3) state <= PACKAGE_WRITE;
                    else begin feature_index <= feature_index + 1; state <= PACKAGE_READ; end
                end
                PACKAGE_WRITE: if (write_req_valid && write_req_ready) begin
                    output_addr_cursor <= output_addr_cursor + 4;
                    if (feature_index + 1 == features_r) state <= WRITE_DRAIN;
                    else begin feature_index <= feature_index + 1; state <= PACKAGE_READ; end
                end
                WRITE_DRAIN: if (!memory_busy) state <= FINISH;
                FINISH: state <= IDLE;
                FAIL: if (!start) state <= IDLE;
                default: state <= IDLE;
            endcase
            if (state == IDLE && start)
                for (clear_lane = 0; clear_lane < MAX_TOKENS; clear_lane = clear_lane + 1) selected[clear_lane] <= 1'b0;
        end
    end
endmodule
