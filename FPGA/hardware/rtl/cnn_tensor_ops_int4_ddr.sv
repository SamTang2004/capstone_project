`timescale 1ns/1ps
// Packed signed-INT4 NCHW tensor operators. Two logical elements occupy each
// DDR byte, with the even element in the low nibble.
module cnn_tensor_ops_int4_ddr (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [2:0]  mode, // 1=maxpool, 2=nearest upsample, 3=concat, 4=avgpool
    input  logic [31:0] input0_base_addr,
    input  logic [31:0] input1_base_addr,
    input  logic [31:0] input2_base_addr,
    input  logic [31:0] output_base_addr,
    input  logic [15:0] n,
    input  logic [15:0] channels0,
    input  logic [15:0] channels1,
    input  logic [15:0] channels2,
    input  logic [15:0] input_h,
    input  logic [15:0] input_w,
    input  logic [15:0] output_h,
    input  logic [15:0] output_w,
    input  logic [7:0]  kernel_h,
    input  logic [7:0]  kernel_w,
    input  logic [7:0]  stride_h,
    input  logic [7:0]  stride_w,
    input  logic [7:0]  pad_h,
    input  logic [7:0]  pad_w,
    output logic        busy,
    output logic        done,
    output logic        error,

    output logic        read_req_valid,
    input  logic        read_req_ready,
    output logic [31:0] read_req_addr,
    input  logic        read_data_valid,
    output logic        read_data_ready,
    input  logic [7:0]  read_data,

    output logic        write_req_valid,
    input  logic        write_req_ready,
    output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data,
    output logic [3:0]  write_req_strb
);
    localparam logic [2:0] MODE_MAXPOOL = 3'd1;
    localparam logic [2:0] MODE_UPSAMPLE = 3'd2;
    localparam logic [2:0] MODE_CONCAT = 3'd3;
    localparam logic [2:0] MODE_AVGPOOL = 3'd4;

    typedef enum logic [4:0] {
        IDLE, VALIDATE, ELEMENT,
        POOL_COORD_MUL, POOL_COORD_OFFSET, POOL_COORD_BOUNDS,
        SOURCE_SELECT, ADDR_NC_MUL, ADDR_NC_ADD, ADDR_H_MUL,
        ADDR_H_ADD, ADDR_W_MUL, ADDR_W_ADD, ADDR_BYTE,
        READ_REQ, READ_WAIT, READ_HANDLE,
        DIV_INIT, DIV_LOOP, PACK, WRITE, FINISH, FAIL
    } state_t;

    state_t state;
    logic [2:0] mode_r;
    logic [31:0] in0_r, in1_r, in2_r, out_r;
    logic [15:0] n_r, c0_r, c1_r, c2_r, total_c_r;
    logic [15:0] in_h_r, in_w_r, out_h_r, out_w_r;
    logic [7:0] kh_r, kw_r, sh_r, sw_r, ph_r, pw_r;

    logic [15:0] n_idx, c_idx, oh_idx, ow_idx;
    logic [7:0] kh_idx, kw_idx;
    logic [15:0] up_src_h, up_src_w;
    logic [7:0] up_h_repeat, up_w_repeat;
    logic [31:0] output_idx_r;

    logic [31:0] source_base_r;
    logic [15:0] source_channels_r, source_channel_r;
    logic [15:0] source_h_r, source_w_r;
    logic signed [31:0] pool_y_r, pool_x_r;
    (* use_dsp = "yes" *) logic [31:0] pool_y_product_r;
    (* use_dsp = "yes" *) logic [31:0] pool_x_product_r;
    (* use_dsp = "yes" *) logic [31:0] nc_product_r;
    (* use_dsp = "yes" *) logic [31:0] h_product_r;
    (* use_dsp = "yes" *) logic [31:0] w_product_r;
    logic [31:0] linear_r, source_addr_r;
    logic source_high_nibble_r;
    logic [7:0] read_byte_r;

    logic signed [3:0] pool_max_r;
    logic signed [31:0] pool_sum_r;
    logic [15:0] pool_count_r;
    logic signed [3:0] element_value_r;
    logic [31:0] packed_word_r, write_addr_r;
    logic [3:0] write_strb_r;
    logic write_last_r;

    logic quotient_negative_r;
    logic [31:0] dividend_r, divisor_r, quotient_r;

    wire pool_mode = mode_r == MODE_MAXPOOL || mode_r == MODE_AVGPOOL;
    wire last_output = n_idx + 1 >= n_r &&
                       c_idx + 1 >= total_c_r &&
                       oh_idx + 1 >= out_h_r &&
                       ow_idx + 1 >= out_w_r;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == READ_REQ;
    assign read_req_addr = source_addr_r;
    assign read_data_ready = state == READ_WAIT;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = packed_word_r;
    assign write_req_strb = write_strb_r;

    task automatic advance_pool_window;
        begin
            if (kw_idx + 1 >= kw_r) begin
                kw_idx <= 0;
                if (kh_idx + 1 >= kh_r) begin
                    kh_idx <= 0;
                    if (mode_r == MODE_AVGPOOL)
                        state <= DIV_INIT;
                    else begin
                        element_value_r <= pool_max_r;
                        state <= PACK;
                    end
                end else begin
                    kh_idx <= kh_idx + 1'b1;
                    state <= POOL_COORD_MUL;
                end
            end else begin
                kw_idx <= kw_idx + 1'b1;
                state <= POOL_COORD_MUL;
            end
        end
    endtask

    task automatic advance_output;
        begin
            if (ow_idx + 1 >= out_w_r) begin
                ow_idx <= 0;
                if (oh_idx + 1 >= out_h_r) begin
                    oh_idx <= 0;
                    if (c_idx + 1 >= total_c_r) begin
                        c_idx <= 0;
                        n_idx <= n_idx + 1'b1;
                    end else begin
                        c_idx <= c_idx + 1'b1;
                    end
                end else begin
                    oh_idx <= oh_idx + 1'b1;
                end
            end else begin
                ow_idx <= ow_idx + 1'b1;
            end

            if (mode_r == MODE_UPSAMPLE) begin
                if (up_w_repeat + 1 >= sw_r) begin
                    up_w_repeat <= 0;
                    up_src_w <= up_src_w + 1'b1;
                end else begin
                    up_w_repeat <= up_w_repeat + 1'b1;
                end
                if (ow_idx + 1 >= out_w_r) begin
                    up_w_repeat <= 0;
                    up_src_w <= 0;
                    if (up_h_repeat + 1 >= sh_r) begin
                        up_h_repeat <= 0;
                        up_src_h <= up_src_h + 1'b1;
                    end else begin
                        up_h_repeat <= up_h_repeat + 1'b1;
                    end
                    if (oh_idx + 1 >= out_h_r) begin
                        up_h_repeat <= 0;
                        up_src_h <= 0;
                    end
                end
            end
        end
    endtask

    always_ff @(posedge clk) begin
        logic signed [3:0] sample;
        logic signed [3:0] next_max;
        logic signed [31:0] next_sum;
        logic [15:0] next_count;
        logic [31:0] next_word;
        logic [3:0] byte_count;
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            error <= 1'b0;
            mode_r <= '0;
            in0_r <= '0; in1_r <= '0; in2_r <= '0; out_r <= '0;
            n_r <= '0; c0_r <= '0; c1_r <= '0; c2_r <= '0;
            total_c_r <= '0;
            in_h_r <= '0; in_w_r <= '0; out_h_r <= '0; out_w_r <= '0;
            kh_r <= '0; kw_r <= '0; sh_r <= '0; sw_r <= '0;
            ph_r <= '0; pw_r <= '0;
            n_idx <= '0; c_idx <= '0; oh_idx <= '0; ow_idx <= '0;
            kh_idx <= '0; kw_idx <= '0;
            up_src_h <= '0; up_src_w <= '0;
            up_h_repeat <= '0; up_w_repeat <= '0;
            output_idx_r <= '0;
            source_base_r <= '0; source_channels_r <= '0;
            source_channel_r <= '0; source_h_r <= '0; source_w_r <= '0;
            pool_y_r <= '0; pool_x_r <= '0;
            pool_y_product_r <= '0; pool_x_product_r <= '0;
            nc_product_r <= '0; h_product_r <= '0; w_product_r <= '0;
            linear_r <= '0; source_addr_r <= '0;
            source_high_nibble_r <= 1'b0;
            read_byte_r <= '0;
            pool_max_r <= -4'sd8; pool_sum_r <= '0; pool_count_r <= '0;
            element_value_r <= '0;
            packed_word_r <= '0; write_addr_r <= '0;
            write_strb_r <= '0; write_last_r <= 1'b0;
            quotient_negative_r <= 1'b0;
            dividend_r <= '0; divisor_r <= '0; quotient_r <= '0;
        end else begin
            done <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        mode_r <= mode;
                        in0_r <= input0_base_addr;
                        in1_r <= input1_base_addr;
                        in2_r <= input2_base_addr;
                        out_r <= output_base_addr;
                        n_r <= n;
                        c0_r <= channels0;
                        c1_r <= channels1;
                        c2_r <= channels2;
                        total_c_r <= mode == MODE_CONCAT
                            ? channels0 + channels1 + channels2 : channels0;
                        in_h_r <= input_h;
                        in_w_r <= input_w;
                        out_h_r <= output_h;
                        out_w_r <= output_w;
                        kh_r <= kernel_h;
                        kw_r <= kernel_w;
                        sh_r <= stride_h;
                        sw_r <= stride_w;
                        ph_r <= pad_h;
                        pw_r <= pad_w;
                        n_idx <= '0; c_idx <= '0; oh_idx <= '0; ow_idx <= '0;
                        kh_idx <= '0; kw_idx <= '0;
                        up_src_h <= '0; up_src_w <= '0;
                        up_h_repeat <= '0; up_w_repeat <= '0;
                        output_idx_r <= '0;
                        packed_word_r <= '0;
                        state <= VALIDATE;
                    end
                end

                VALIDATE: begin
                    if (mode_r < MODE_MAXPOOL || mode_r > MODE_AVGPOOL ||
                        n_r == 0 || c0_r == 0 || in_h_r == 0 || in_w_r == 0 ||
                        out_h_r == 0 || out_w_r == 0 || out_r[1:0] != 0 ||
                        (pool_mode && (kh_r == 0 || kw_r == 0 ||
                                       sh_r == 0 || sw_r == 0)) ||
                        (mode_r == MODE_UPSAMPLE &&
                            (sh_r == 0 || sw_r == 0 ||
                             out_h_r != in_h_r * sh_r ||
                             out_w_r != in_w_r * sw_r))) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        state <= ELEMENT;
                    end
                end

                ELEMENT: begin
                    if (pool_mode) begin
                        kh_idx <= 0;
                        kw_idx <= 0;
                        pool_max_r <= -4'sd8;
                        pool_sum_r <= 0;
                        pool_count_r <= 0;
                        state <= POOL_COORD_MUL;
                    end else begin
                        state <= SOURCE_SELECT;
                    end
                end

                POOL_COORD_MUL: begin
                    pool_y_product_r <= oh_idx * sh_r;
                    pool_x_product_r <= ow_idx * sw_r;
                    state <= POOL_COORD_OFFSET;
                end

                POOL_COORD_OFFSET: begin
                    pool_y_r <= $signed({1'b0, pool_y_product_r}) +
                                $signed({1'b0, kh_idx}) -
                                $signed({1'b0, ph_r});
                    pool_x_r <= $signed({1'b0, pool_x_product_r}) +
                                $signed({1'b0, kw_idx}) -
                                $signed({1'b0, pw_r});
                    state <= POOL_COORD_BOUNDS;
                end

                POOL_COORD_BOUNDS: begin
                    if (pool_y_r < 0 || pool_x_r < 0 ||
                        pool_y_r >= $signed({1'b0, in_h_r}) ||
                        pool_x_r >= $signed({1'b0, in_w_r})) begin
                        advance_pool_window();
                    end else begin
                        state <= SOURCE_SELECT;
                    end
                end

                SOURCE_SELECT: begin
                    source_h_r <= pool_mode ? pool_y_r[15:0] :
                                    (mode_r == MODE_UPSAMPLE ? up_src_h : oh_idx);
                    source_w_r <= pool_mode ? pool_x_r[15:0] :
                                    (mode_r == MODE_UPSAMPLE ? up_src_w : ow_idx);
                    if (mode_r != MODE_CONCAT || c_idx < c0_r) begin
                        source_base_r <= in0_r;
                        source_channels_r <= c0_r;
                        source_channel_r <= c_idx;
                    end else if (c_idx < c0_r + c1_r) begin
                        source_base_r <= in1_r;
                        source_channels_r <= c1_r;
                        source_channel_r <= c_idx - c0_r;
                    end else begin
                        source_base_r <= in2_r;
                        source_channels_r <= c2_r;
                        source_channel_r <= c_idx - c0_r - c1_r;
                    end
                    state <= ADDR_NC_MUL;
                end

                ADDR_NC_MUL: begin
                    nc_product_r <= n_idx * source_channels_r;
                    state <= ADDR_NC_ADD;
                end
                ADDR_NC_ADD: begin
                    linear_r <= nc_product_r + source_channel_r;
                    state <= ADDR_H_MUL;
                end
                ADDR_H_MUL: begin
                    h_product_r <= linear_r * in_h_r;
                    state <= ADDR_H_ADD;
                end
                ADDR_H_ADD: begin
                    linear_r <= h_product_r + source_h_r;
                    state <= ADDR_W_MUL;
                end
                ADDR_W_MUL: begin
                    w_product_r <= linear_r * in_w_r;
                    state <= ADDR_W_ADD;
                end
                ADDR_W_ADD: begin
                    linear_r <= w_product_r + source_w_r;
                    state <= ADDR_BYTE;
                end
                ADDR_BYTE: begin
                    source_addr_r <= source_base_r + (linear_r >> 1);
                    source_high_nibble_r <= linear_r[0];
                    state <= READ_REQ;
                end

                READ_REQ: if (read_req_ready) state <= READ_WAIT;
                READ_WAIT: if (read_data_valid) begin
                    read_byte_r <= read_data;
                    state <= READ_HANDLE;
                end

                READ_HANDLE: begin
                    sample = source_high_nibble_r
                        ? $signed(read_byte_r[7:4])
                        : $signed(read_byte_r[3:0]);
                    if (mode_r == MODE_MAXPOOL) begin
                        next_max = sample > pool_max_r ? sample : pool_max_r;
                        pool_max_r <= next_max;
                        if (kh_idx + 1 >= kh_r && kw_idx + 1 >= kw_r) begin
                            kh_idx <= 0;
                            kw_idx <= 0;
                            element_value_r <= next_max;
                            state <= PACK;
                        end else begin
                            advance_pool_window();
                        end
                    end else if (mode_r == MODE_AVGPOOL) begin
                        next_sum = pool_sum_r + sample;
                        next_count = pool_count_r + 1'b1;
                        pool_sum_r <= next_sum;
                        pool_count_r <= next_count;
                        if (kh_idx + 1 >= kh_r && kw_idx + 1 >= kw_r) begin
                            kh_idx <= 0;
                            kw_idx <= 0;
                            state <= DIV_INIT;
                        end else begin
                            advance_pool_window();
                        end
                    end else begin
                        element_value_r <= sample;
                        state <= PACK;
                    end
                end

                DIV_INIT: begin
                    if (pool_count_r == 0) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        quotient_negative_r <= pool_sum_r < 0;
                        dividend_r <= pool_sum_r < 0 ? -pool_sum_r : pool_sum_r;
                        divisor_r <= pool_count_r;
                        quotient_r <= 0;
                        state <= DIV_LOOP;
                    end
                end

                DIV_LOOP: begin
                    if (dividend_r >= divisor_r) begin
                        dividend_r <= dividend_r - divisor_r;
                        quotient_r <= quotient_r + 1'b1;
                    end else begin
                        element_value_r <= quotient_negative_r
                            ? -$signed(quotient_r[3:0])
                            : $signed(quotient_r[3:0]);
                        state <= PACK;
                    end
                end

                PACK: begin
                    next_word = packed_word_r;
                    next_word[output_idx_r[2:0]*4 +: 4] = element_value_r;
                    packed_word_r <= next_word;
                    if (output_idx_r[2:0] == 3'd7 || last_output) begin
                        byte_count = (output_idx_r[2:0] + 2) >> 1;
                        write_addr_r <= out_r + ((output_idx_r >> 3) << 2);
                        write_strb_r <= output_idx_r[2:0] == 3'd7
                            ? 4'hf : ((4'b0001 << byte_count) - 1'b1);
                        write_last_r <= last_output;
                        packed_word_r <= next_word;
                        state <= WRITE;
                    end else begin
                        output_idx_r <= output_idx_r + 1'b1;
                        advance_output();
                        state <= ELEMENT;
                    end
                end

                WRITE: if (write_req_ready) begin
                    packed_word_r <= '0;
                    if (write_last_r) begin
                        state <= FINISH;
                    end else begin
                        output_idx_r <= output_idx_r + 1'b1;
                        advance_output();
                        state <= ELEMENT;
                    end
                end

                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
