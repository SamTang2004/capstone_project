module transformer_rowwise_ddr #(
    parameter int MAX_COLUMNS = 1024
) (
    input  logic clk, rst_n, pipeline_enable, start, softmax_mode, output_tiled,
    input  logic [31:0] input_base_addr, gamma_base_addr, beta_base_addr,
    input  logic [31:0] output_base_addr, epsilon_term,
    input  logic [15:0] rows, columns,
    output logic busy, done, error, op_start, op_done,
    output logic memory_stall, input_stall, output_stall,
    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [6:0] read_req_len,
    output logic read_req_wide, read_req_qword, read_req_allow_ahead,
    input  logic read_data_valid, output logic read_data_ready,
    input  logic [7:0] read_data, input logic read_data_last,
    output logic write_req_valid, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic [31:0] write_req_data,
    output logic [3:0] write_req_strb, output logic write_req_last,
    input  logic memory_busy, memory_error
);
    typedef enum logic [4:0] {
        IDLE, PARAM_G_PREP, PARAM_G_REQ, PARAM_G_RECV,
        PARAM_B_PREP, PARAM_B_REQ, PARAM_B_RECV,
        ROW_START, IN_PREP, IN_REQ, IN_RECV, OUTPUT,
        WRITE_ARM, WRITE_WAIT, FINISH, FAIL
    } state_t;
    state_t state;

    logic mode_r, tiled_r;
    logic [15:0] rows_r, cols_r, row_idx, col_idx;
    logic [31:0] eps_r, input_addr_r, byte_output_addr;
    (* extract_enable = "no" *) logic [31:0] output_row_base_r;
    logic [31:0] gamma_base_r, beta_base_r;
    logic [16:0] param_byte_index, param_total_bytes;
    logic [7:0] param_low;
    logic [6:0] response_bytes_left;
    logic [31:0] read_addr_r;
    logic [6:0] read_len_r;
    (* ram_style = "block" *) logic signed [15:0] gamma_mem [0:MAX_COLUMNS-1];
    (* ram_style = "block" *) logic signed [15:0] beta_mem [0:MAX_COLUMNS-1];
    logic signed [15:0] gamma_r, beta_r;
    logic [9:0] param_read_addr;
    logic gamma_write_en, beta_write_en;

    logic engine_start, engine_input_valid, engine_input_ready;
    logic engine_param_valid, engine_param_ready;
    logic engine_output_valid, engine_output_last, engine_busy, engine_done, engine_error;
    logic signed [7:0] engine_output_data;
    logic [7:0] output_byte;

    function automatic logic [6:0] request_size(
        input logic [16:0] remaining,
        input logic [31:0] address
    );
        logic [6:0] line_remaining;
        begin
            line_remaining = 7'd64 - {1'b0, address[5:0]};
            request_size = remaining < line_remaining ? remaining[6:0] : line_remaining;
        end
    endfunction

    rowwise_norm_softmax_engine #(.MAX_LENGTH(MAX_COLUMNS)) engine (
        .clk, .rst_n, .start(engine_start), .softmax_mode(mode_r),
        .vector_length(cols_r), .epsilon_term(eps_r),
        .input_valid(engine_input_valid), .input_ready(engine_input_ready), .input_data(read_data),
        .param_valid(engine_param_valid), .param_ready(engine_param_ready),
        .gamma_q8(gamma_r), .beta_q8(beta_r),
        .output_valid(engine_output_valid), .output_ready(write_req_ready && state == OUTPUT),
        .output_data(engine_output_data), .output_last(engine_output_last),
        .busy(engine_busy), .done(engine_done), .error(engine_error)
    );

    assign engine_start = state == ROW_START;
    assign engine_input_valid = state == IN_RECV && read_data_valid;
    assign engine_param_valid = state == OUTPUT && !mode_r;
    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign op_start = state == IDLE && start;
    assign op_done = state == FINISH;

    assign read_req_valid = pipeline_enable &&
        (state == PARAM_G_REQ || state == PARAM_B_REQ || state == IN_REQ);
    assign read_req_addr = read_addr_r;
    assign read_req_len = read_len_r;
    assign read_req_wide = 1'b0;
    assign read_req_qword = 1'b0;
    assign read_req_allow_ahead = 1'b0;
    assign read_data_ready = pipeline_enable &&
        ((state == PARAM_G_RECV || state == PARAM_B_RECV) ||
         (state == IN_RECV && engine_input_ready));

    assign output_byte = engine_output_data;
    assign write_req_valid = pipeline_enable && state == OUTPUT && engine_output_valid;
    assign write_req_addr = {byte_output_addr[31:2], 2'b0};
    assign write_req_data = {4{output_byte}} << (byte_output_addr[1:0] * 8);
    assign write_req_strb = 4'b1 << byte_output_addr[1:0];
    assign write_req_last = 1'b1;
    assign memory_stall = busy && ((read_req_valid && !read_req_ready) ||
                                   (read_data_ready && !read_data_valid) ||
                                   (write_req_valid && !write_req_ready));
    assign input_stall = read_data_ready && !read_data_valid;
    assign output_stall = write_req_valid && !write_req_ready;

    assign gamma_write_en = pipeline_enable && state == PARAM_G_RECV &&
        read_data_valid && read_data_ready && param_byte_index[0];
    assign beta_write_en = pipeline_enable && state == PARAM_B_RECV &&
        read_data_valid && read_data_ready && param_byte_index[0];
    // The BRAM read is synchronous.  On an accepted output, prefetch the next
    // column at the same edge that col_idx advances; otherwise LN_CENTER would
    // see the previous column's gamma/beta for every element after column 0.
    assign param_read_addr = state == OUTPUT && write_req_valid && write_req_ready &&
        !engine_output_last ? col_idx[9:0] + 1'b1 : col_idx[9:0];

    // Canonical synchronous simple-dual-port pattern keeps both parameter
    // vectors in BRAM instead of expanding them into thousands of registers.
    always_ff @(posedge clk) begin
        if (gamma_write_en)
            gamma_mem[param_byte_index[10:1]] <= {read_data, param_low};
        gamma_r <= gamma_mem[param_read_addr];
    end
    always_ff @(posedge clk) begin
        if (beta_write_en)
            beta_mem[param_byte_index[10:1]] <= {read_data, param_low};
        beta_r <= beta_mem[param_read_addr];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; error <= 1'b0;
            mode_r <= 1'b0; tiled_r <= 1'b0; rows_r <= '0; cols_r <= '0;
            row_idx <= '0; col_idx <= '0; eps_r <= '0;
            input_addr_r <= '0; output_row_base_r <= '0; byte_output_addr <= '0;
            gamma_base_r <= '0; beta_base_r <= '0;
            param_byte_index <= '0; param_total_bytes <= '0; param_low <= '0;
            response_bytes_left <= '0; read_addr_r <= '0; read_len_r <= '0;
        end else begin
            done <= 1'b0;
            if (memory_error || engine_error) begin
                error <= 1'b1;
                state <= FAIL;
            end else if (pipeline_enable) unique case (state)
                IDLE: if (start) begin
                    error <= 1'b0;
                    if (rows == 0 || columns == 0 || columns > MAX_COLUMNS ||
                        (softmax_mode && columns > 256)) begin
                        error <= 1'b1;
                        state <= FAIL;
                    end else begin
                        mode_r <= softmax_mode; tiled_r <= output_tiled;
                        rows_r <= rows; cols_r <= columns; eps_r <= epsilon_term;
                        row_idx <= 0; col_idx <= 0; input_addr_r <= input_base_addr;
                        output_row_base_r <= output_base_addr;
                        gamma_base_r <= gamma_base_addr; beta_base_r <= beta_base_addr;
                        param_total_bytes <= columns << 1; param_byte_index <= 0;
                        state <= softmax_mode ? ROW_START : PARAM_G_PREP;
                    end
                end

                PARAM_G_PREP: begin
                    read_addr_r <= gamma_base_r + param_byte_index;
                    read_len_r <= request_size(param_total_bytes - param_byte_index,
                                               gamma_base_r + param_byte_index);
                    state <= PARAM_G_REQ;
                end
                PARAM_G_REQ: if (read_req_valid && read_req_ready) begin
                    response_bytes_left <= read_len_r;
                    state <= PARAM_G_RECV;
                end
                PARAM_G_RECV: if (read_data_valid && read_data_ready) begin
                    if (read_data_last != (response_bytes_left == 1)) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        response_bytes_left <= response_bytes_left - 1'b1;
                        if (!param_byte_index[0]) param_low <= read_data;
                        if (param_byte_index + 1'b1 == param_total_bytes) begin
                            param_byte_index <= 0;
                            state <= PARAM_B_PREP;
                        end else begin
                            param_byte_index <= param_byte_index + 1'b1;
                            if (read_data_last) state <= PARAM_G_PREP;
                        end
                    end
                end

                PARAM_B_PREP: begin
                    read_addr_r <= beta_base_r + param_byte_index;
                    read_len_r <= request_size(param_total_bytes - param_byte_index,
                                               beta_base_r + param_byte_index);
                    state <= PARAM_B_REQ;
                end
                PARAM_B_REQ: if (read_req_valid && read_req_ready) begin
                    response_bytes_left <= read_len_r;
                    state <= PARAM_B_RECV;
                end
                PARAM_B_RECV: if (read_data_valid && read_data_ready) begin
                    if (read_data_last != (response_bytes_left == 1)) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        response_bytes_left <= response_bytes_left - 1'b1;
                        if (!param_byte_index[0]) param_low <= read_data;
                        if (param_byte_index + 1'b1 == param_total_bytes) begin
                            param_byte_index <= 0;
                            state <= ROW_START;
                        end else begin
                            param_byte_index <= param_byte_index + 1'b1;
                            if (read_data_last) state <= PARAM_B_PREP;
                        end
                    end
                end

                ROW_START: begin
                    col_idx <= 0;
                    byte_output_addr <= output_row_base_r;
                    state <= IN_PREP;
                end
                IN_PREP: begin
                    read_addr_r <= input_addr_r;
                    read_len_r <= request_size(cols_r - col_idx, input_addr_r);
                    state <= IN_REQ;
                end
                IN_REQ: if (read_req_valid && read_req_ready) begin
                    response_bytes_left <= read_len_r;
                    state <= IN_RECV;
                end
                IN_RECV: if (read_data_valid && read_data_ready) begin
                    if (read_data_last != (response_bytes_left == 1)) begin
                        error <= 1'b1; state <= FAIL;
                    end else begin
                        response_bytes_left <= response_bytes_left - 1'b1;
                        input_addr_r <= input_addr_r + 1'b1;
                        if (col_idx + 1'b1 == cols_r) begin
                            col_idx <= 0;
                            state <= OUTPUT;
                        end else begin
                            col_idx <= col_idx + 1'b1;
                            if (read_data_last) state <= IN_PREP;
                        end
                    end
                end

                OUTPUT: if (write_req_valid && write_req_ready) begin
                    if (engine_output_last) begin
                        if (row_idx + 1'b1 == rows_r) state <= WRITE_ARM;
                        else begin
                            row_idx <= row_idx + 1'b1;
                            if (tiled_r) begin
                                if (row_idx[2:0] == 3'd7)
                                    output_row_base_r <= output_row_base_r + (cols_r << 3) - 7;
                                else output_row_base_r <= output_row_base_r + 1'b1;
                            end else output_row_base_r <= byte_output_addr + 1'b1;
                            state <= ROW_START;
                        end
                    end else begin
                        col_idx <= col_idx + 1'b1;
                        byte_output_addr <= byte_output_addr + (tiled_r ? 8 : 1);
                    end
                end
                WRITE_ARM: state <= WRITE_WAIT;
                WRITE_WAIT: if (!memory_busy) state <= FINISH;
                FINISH: begin done <= 1'b1; state <= IDLE; end
                FAIL: begin done <= 1'b1; if (!start) state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
