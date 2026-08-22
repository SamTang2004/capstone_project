// DDR-resident INT8 CNN tensor operators.  Output is produced in NCHW order,
// so packed writes remain contiguous and the memory I/O driver can coalesce them.
module cnn_tensor_ops_ddr (
    input logic clk, input logic rst_n, input logic start,
    input logic [1:0] mode, // 1=max-pool, 2=nearest upsample, 3=channel concat
    input logic [31:0] input0_base_addr, input logic [31:0] input1_base_addr,
    input logic [31:0] input2_base_addr, input logic [31:0] output_base_addr,
    input logic [15:0] n, input logic [15:0] channels0, input logic [15:0] channels1, input logic [15:0] channels2,
    input logic [15:0] input_h, input logic [15:0] input_w, input logic [15:0] output_h, input logic [15:0] output_w,
    input logic [7:0] kernel_h, input logic [7:0] kernel_w, input logic [7:0] stride_h, input logic [7:0] stride_w,
    input logic [7:0] pad_h, input logic [7:0] pad_w,
    output logic busy, output logic done, output logic error,
    output logic read_req_valid, input logic read_req_ready, output logic [31:0] read_req_addr,
    output logic [4:0] read_req_len,
    input logic read_word_valid, output logic read_word_ready,
    input logic [31:0] read_word_data, input logic [3:0] read_word_keep,
    input logic read_word_last,
    output logic write_req_valid, input logic write_req_ready, output logic [31:0] write_req_addr,
    output logic [31:0] write_req_data, output logic [3:0] write_req_strb
);
    localparam logic [1:0] MODE_POOL = 2'd1, MODE_UPSAMPLE = 2'd2, MODE_CONCAT = 2'd3;
    typedef enum logic [4:0] {
        IDLE, VALIDATE_CALC, VALIDATE_DECIDE, ELEMENT, ELEMENT_FLAGS,
        POOL_COORD_MUL, POOL_COORD_OFFSET, POOL_COORD_BOUNDS,
        ADDR_SELECT, ADDR_COORD_CAPTURE, ADDR_BATCH_OPERANDS,
        ADDR_BATCH_MUL, ADDR_CHANNEL_ADD,
        ADDR_ROW_MUL, ADDR_ROW_ADD, ADDR_COL_MUL, ADDR_BASE_ADD,
        ADDR_FINAL_ADD, POOL_REQUEST, POOL_RECEIVE, POOL_REDUCE,
        PACK, WRITE, FINISH, FAIL
    } state_t;
    state_t state;
    logic [1:0] mode_r;
    logic [31:0] in0_r, in1_r, in2_r, out_r, output_index_r;
    logic [15:0] n_r, c0_r, c1_r, c2_r, c01_r, total_channels_r, in_h_r, in_w_r, out_h_r, out_w_r;
    logic [7:0] kh_size_r, kw_size_r, sh_r, sw_r, ph_r, pw_r;
    logic [15:0] n_idx, c_idx, oh_idx, ow_idx;
    logic [7:0] kh_idx, kw_idx;
    logic signed [7:0] pool_max_r;
    logic signed [7:0] pool_word_max_r;
    logic [7:0] element_value_r;
    logic [31:0] packed_word_r, write_data_r, write_addr_r;
    logic [3:0] write_strb_r;
    logic write_last_r, last_element_r;
    logic last_n_r, last_c_r, last_oh_r, last_ow_r;
    logic [7:0] up_h_repeat, up_w_repeat;
    logic [15:0] up_src_h, up_src_w;
    logic [23:0] pool_h_product_r, pool_w_product_r;
    logic signed [31:0] pool_ih_r, pool_iw_r;
    logic pool_in_bounds_r;
    logic [31:0] source_base_r, address_index_r;
    // Keep final byte-address adds in fabric. Packing them into a DSP made
    // its clock-enable depend on the source-coordinate selection cone.
    (* use_dsp = "no" *) logic [31:0] source_addr_r, address_base_r;
    logic [15:0] source_channels_r, source_channel_offset_r;
    // Isolate the DSP operand/clock-enable cone from bank and coordinate
    // selection.  The `keep` boundary prevents Vivado from folding the
    // selection muxes back into the DSP enable path.
    (* keep = "true" *) logic [15:0] batch_n_r, batch_channels_r, batch_channel_offset_r;
    (* use_dsp = "yes" *) logic [31:0] batch_channel_product_r;
    (* use_dsp = "yes" *) logic [31:0] row_product_r;
    (* use_dsp = "yes" *) logic [31:0] column_product_r;
    logic [15:0] source_h_r, source_w_r;
    logic validation_mode_error_r, validation_shape_error_r;
    logic validation_pool_error_r, validation_upsample_error_r;
    logic [23:0] validation_up_h_r, validation_up_w_r;
    logic [2:0] read_run_len_r;
    wire [2:0] response_bytes = read_word_keep[0] + read_word_keep[1] +
                                read_word_keep[2] + read_word_keep[3];
    wire signed [7:0] pool_lane0 = $signed(read_word_data[7:0]);
    wire signed [7:0] pool_lane1 = read_run_len_r > 1 ? $signed(read_word_data[15:8]) : -8'sd128;
    wire signed [7:0] pool_lane2 = read_run_len_r > 2 ? $signed(read_word_data[23:16]) : -8'sd128;
    wire signed [7:0] pool_lane3 = read_run_len_r > 3 ? $signed(read_word_data[31:24]) : -8'sd128;
    wire signed [7:0] pool_pair01_max = pool_lane0 > pool_lane1 ? pool_lane0 : pool_lane1;
    wire signed [7:0] pool_pair23_max = pool_lane2 > pool_lane3 ? pool_lane2 : pool_lane3;
    wire signed [7:0] pool_response_max = pool_pair01_max > pool_pair23_max ? pool_pair01_max : pool_pair23_max;
    wire signed [7:0] pool_merged_max = pool_word_max_r > pool_max_r ? pool_word_max_r : pool_max_r;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign read_req_valid = state == POOL_REQUEST;
    assign read_req_len = {2'd0, read_run_len_r};
    assign read_word_ready = state == POOL_RECEIVE;
    assign write_req_valid = state == WRITE;
    assign write_req_addr = write_addr_r;
    assign write_req_data = write_data_r;
    assign write_req_strb = write_strb_r;

    assign read_req_addr = source_addr_r;
    task automatic advance_output;
        begin
            if (last_ow_r) begin
                ow_idx <= 0;
                if (last_oh_r) begin
                    oh_idx <= 0;
                    if (last_c_r) begin
                        c_idx <= 0;
                        n_idx <= n_idx + 1'b1;
                    end else c_idx <= c_idx + 1'b1;
                end else oh_idx <= oh_idx + 1'b1;
            end else ow_idx <= ow_idx + 1'b1;

            if (mode_r == MODE_UPSAMPLE) begin
                if (up_w_repeat + 1 >= sw_r) begin
                    up_w_repeat <= 0;
                    up_src_w <= up_src_w + 1'b1;
                end else up_w_repeat <= up_w_repeat + 1'b1;
                if (last_ow_r) begin
                    up_w_repeat <= 0;
                    up_src_w <= 0;
                    if (up_h_repeat + 1 >= sh_r) begin
                        up_h_repeat <= 0;
                        up_src_h <= up_src_h + 1'b1;
                    end else up_h_repeat <= up_h_repeat + 1'b1;
                    if (last_oh_r) begin
                        up_h_repeat <= 0;
                        up_src_h <= 0;
                    end
                end
            end
        end
    endtask

    task automatic advance_pool_window;
        begin
            if (kw_idx + 1 >= kw_size_r) begin
                kw_idx <= 0;
                if (kh_idx + 1 >= kh_size_r) begin
                    kh_idx <= 0;
                    element_value_r <= pool_max_r;
                    state <= PACK;
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

    always_ff @(posedge clk) begin
        logic [31:0] next_packed;
        logic [3:0] final_strobe;
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            mode_r <= 0; in0_r <= 0; in1_r <= 0; in2_r <= 0; out_r <= 0; output_index_r <= 0;
            n_r <= 0; c0_r <= 0; c1_r <= 0; c2_r <= 0; c01_r <= 0; total_channels_r <= 0; in_h_r <= 0; in_w_r <= 0; out_h_r <= 0; out_w_r <= 0;
            kh_size_r <= 0; kw_size_r <= 0; sh_r <= 0; sw_r <= 0; ph_r <= 0; pw_r <= 0;
            n_idx <= 0; c_idx <= 0; oh_idx <= 0; ow_idx <= 0; kh_idx <= 0; kw_idx <= 0;
            pool_max_r <= -8'sd128; pool_word_max_r <= -8'sd128;
            packed_word_r <= 0; write_data_r <= 0; write_addr_r <= 0; write_strb_r <= 0; write_last_r <= 0; up_h_repeat <= 0; up_w_repeat <= 0; up_src_h <= 0; up_src_w <= 0;
            source_base_r <= 0; source_addr_r <= 0; address_index_r <= 0; address_base_r <= 0;
            source_channels_r <= 0; source_channel_offset_r <= 0; batch_channel_product_r <= 0; row_product_r <= 0; column_product_r <= 0;
            batch_n_r <= 0; batch_channels_r <= 0; batch_channel_offset_r <= 0;
            source_h_r <= 0; source_w_r <= 0;
            pool_h_product_r <= 0; pool_w_product_r <= 0;
            pool_ih_r <= 0; pool_iw_r <= 0; pool_in_bounds_r <= 1'b0;
            last_n_r <= 0; last_c_r <= 0; last_oh_r <= 0; last_ow_r <= 0; last_element_r <= 0;
            validation_mode_error_r <= 0; validation_shape_error_r <= 0;
            validation_pool_error_r <= 0; validation_upsample_error_r <= 0;
            validation_up_h_r <= 0; validation_up_w_r <= 0;
            read_run_len_r <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        mode_r <= mode; in0_r <= input0_base_addr; in1_r <= input1_base_addr; in2_r <= input2_base_addr; out_r <= output_base_addr;
                        n_r <= n; c0_r <= channels0; c1_r <= channels1; c2_r <= channels2;
                        c01_r <= channels0 + channels1; total_channels_r <= mode == MODE_CONCAT ? channels0 + channels1 + channels2 : channels0;
                        in_h_r <= input_h; in_w_r <= input_w; out_h_r <= output_h; out_w_r <= output_w;
                        kh_size_r <= kernel_h; kw_size_r <= kernel_w; sh_r <= stride_h; sw_r <= stride_w; ph_r <= pad_h; pw_r <= pad_w;
                        output_index_r <= 0; n_idx <= 0; c_idx <= 0; oh_idx <= 0; ow_idx <= 0; packed_word_r <= 0;
                        kh_idx <= 0; kw_idx <= 0; up_h_repeat <= 0; up_w_repeat <= 0; up_src_h <= 0; up_src_w <= 0;
                        state <= VALIDATE_CALC;
                    end
                end
                VALIDATE_CALC: begin
                    validation_mode_error_r <= mode_r < MODE_POOL || mode_r > MODE_CONCAT;
                    validation_shape_error_r <= n_r == 0 || c0_r == 0 || in_h_r == 0 || in_w_r == 0 || out_h_r == 0 || out_w_r == 0;
                    validation_pool_error_r <= mode_r == MODE_POOL && (kh_size_r == 0 || kw_size_r == 0 || sh_r == 0 || sw_r == 0);
                    validation_upsample_error_r <= mode_r == MODE_UPSAMPLE && (sh_r == 0 || sw_r == 0);
                    validation_up_h_r <= in_h_r * sh_r;
                    validation_up_w_r <= in_w_r * sw_r;
                    state <= VALIDATE_DECIDE;
                end
                VALIDATE_DECIDE: begin
                    if (validation_mode_error_r || validation_shape_error_r || validation_pool_error_r || validation_upsample_error_r ||
                        (mode_r == MODE_UPSAMPLE && (out_h_r != validation_up_h_r || out_w_r != validation_up_w_r))) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else state <= ELEMENT;
                end
                ELEMENT: begin
                    if (mode_r == MODE_POOL) begin
                        kh_idx <= 0; kw_idx <= 0; pool_max_r <= -8'sd128;
                    end
                    last_n_r <= n_idx + 1'b1 >= n_r;
                    last_c_r <= c_idx + 1'b1 >= total_channels_r;
                    last_oh_r <= oh_idx + 1'b1 >= out_h_r;
                    last_ow_r <= ow_idx + 1'b1 >= out_w_r;
                    state <= ELEMENT_FLAGS;
                end
                ELEMENT_FLAGS: begin
                    last_element_r <= last_n_r && last_c_r && last_oh_r && last_ow_r;
                    state <= mode_r == MODE_POOL ? POOL_COORD_MUL : ADDR_SELECT;
                end
                POOL_COORD_MUL: begin
                    pool_h_product_r <= oh_idx * sh_r;
                    pool_w_product_r <= ow_idx * sw_r;
                    state <= POOL_COORD_OFFSET;
                end
                POOL_COORD_OFFSET: begin
                    pool_ih_r <= $signed({1'b0, pool_h_product_r}) + $signed({1'b0, kh_idx}) - $signed({1'b0, ph_r});
                    pool_iw_r <= $signed({1'b0, pool_w_product_r}) + $signed({1'b0, kw_idx}) - $signed({1'b0, pw_r});
                    state <= POOL_COORD_BOUNDS;
                end
                POOL_COORD_BOUNDS: begin
                    pool_in_bounds_r <= pool_ih_r >= 0 && pool_iw_r >= 0 &&
                        pool_ih_r < $signed({1'b0, in_h_r}) && pool_iw_r < $signed({1'b0, in_w_r});
                    state <= ADDR_SELECT;
                end
                // Pipeline NCHW byte address generation.  The old ADDR0 stage
                // combined concat selection, multiplication, subtraction, and
                // addition into one 29-level path.  Each arithmetic result is
                // now registered before feeding the next stage.
                ADDR_SELECT: begin
                    if (mode_r == MODE_POOL && !pool_in_bounds_r) begin
                        advance_pool_window();
                    end else begin
                        if (mode_r == MODE_POOL || mode_r == MODE_UPSAMPLE) begin
                            source_base_r <= in0_r; source_channels_r <= c0_r; source_channel_offset_r <= c_idx;
                        end else if (c_idx < c0_r) begin
                            source_base_r <= in0_r; source_channels_r <= c0_r; source_channel_offset_r <= c_idx;
                        end else if (c_idx < c01_r) begin
                            source_base_r <= in1_r; source_channels_r <= c1_r; source_channel_offset_r <= c_idx - c0_r;
                        end else begin
                            source_base_r <= in2_r; source_channels_r <= c2_r; source_channel_offset_r <= c_idx - c01_r;
                        end
                        state <= ADDR_COORD_CAPTURE;
                    end
                end
                // Isolate output/pool coordinates from the address-bank
                // selection and subsequent DSP-backed multipliers.
                ADDR_COORD_CAPTURE: begin
                    if (mode_r == MODE_POOL) begin
                        source_h_r <= pool_ih_r;
                        source_w_r <= pool_iw_r;
                    end else if (mode_r == MODE_UPSAMPLE) begin
                        source_h_r <= up_src_h;
                        source_w_r <= up_src_w;
                    end else begin
                        source_h_r <= oh_idx;
                        source_w_r <= ow_idx;
                    end
                    state <= ADDR_BATCH_OPERANDS;
                end
                ADDR_BATCH_OPERANDS: begin
                    batch_n_r <= n_idx;
                    batch_channels_r <= source_channels_r;
                    batch_channel_offset_r <= source_channel_offset_r;
                    state <= ADDR_BATCH_MUL;
                end
                ADDR_BATCH_MUL: begin
                    batch_channel_product_r <= batch_n_r * batch_channels_r;
                    state <= ADDR_CHANNEL_ADD;
                end
                ADDR_CHANNEL_ADD: begin
                    address_index_r <= batch_channel_product_r + batch_channel_offset_r;
                    state <= ADDR_ROW_MUL;
                end
                ADDR_ROW_MUL: begin
                    row_product_r <= address_index_r * in_h_r;
                    state <= ADDR_ROW_ADD;
                end
                ADDR_ROW_ADD: begin
                    address_index_r <= row_product_r + source_h_r;
                    state <= ADDR_COL_MUL;
                end
                ADDR_COL_MUL: begin
                    column_product_r <= address_index_r * in_w_r;
                    state <= ADDR_BASE_ADD;
                end
                ADDR_BASE_ADD: begin
                    address_base_r <= source_base_r + column_product_r;
                    state <= ADDR_FINAL_ADD;
                end
                ADDR_FINAL_ADD: begin
                    source_addr_r <= address_base_r + source_w_r;
                    if (mode_r != MODE_POOL) read_run_len_r <= 3'd1;
                    else if (kw_size_r - kw_idx >= 4 && in_w_r - pool_iw_r >= 4) read_run_len_r <= 3'd4;
                    else if (kw_size_r - kw_idx < in_w_r - pool_iw_r) read_run_len_r <= kw_size_r - kw_idx;
                    else read_run_len_r <= in_w_r - pool_iw_r;
                    state <= POOL_REQUEST;
                end
                POOL_REQUEST: if (read_req_ready) state <= POOL_RECEIVE;
                POOL_RECEIVE: if (read_word_valid) begin
                    if (response_bytes != read_run_len_r ||
                        read_word_keep != ((5'd1 << read_run_len_r) - 1'b1) ||
                        !read_word_last) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else if (mode_r == MODE_POOL) begin
                        pool_word_max_r <= pool_response_max;
                        state <= POOL_REDUCE;
                    end else begin
                        element_value_r <= read_word_data[7:0];
                        state <= PACK;
                    end
                end
                POOL_REDUCE: begin
                        pool_max_r <= pool_merged_max;
                        if (kw_idx + read_run_len_r >= kw_size_r) begin
                            kw_idx <= 0;
                            if (kh_idx + 1 >= kh_size_r) begin
                                element_value_r <= pool_merged_max;
                                kh_idx <= 0;
                                state <= PACK;
                            end else begin
                                kh_idx <= kh_idx + 1'b1;
                                state <= POOL_COORD_MUL;
                            end
                        end else begin
                            kw_idx <= kw_idx + read_run_len_r;
                            state <= POOL_COORD_MUL;
                        end
                end
                PACK: begin
                    next_packed = packed_word_r;
                    next_packed[output_index_r[1:0] * 8 +: 8] = element_value_r;
                    packed_word_r <= next_packed;
                    if (output_index_r[1:0] == 2'd3 || last_element_r) begin
                        final_strobe = output_index_r[1:0] == 2'd3 ? 4'hf : ((4'b0001 << (output_index_r[1:0] + 1'b1)) - 1'b1);
                        write_data_r <= next_packed; write_addr_r <= out_r + (output_index_r & 32'hffff_fffc); write_strb_r <= final_strobe; write_last_r <= last_element_r; state <= WRITE;
                    end else begin
                        output_index_r <= output_index_r + 1'b1; advance_output(); state <= ELEMENT;
                    end
                end
                WRITE: if (write_req_ready) begin
                    packed_word_r <= 0;
                    if (write_last_r) state <= FINISH;
                    else begin output_index_r <= output_index_r + 1'b1; advance_output(); state <= ELEMENT; end
                end
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
