`timescale 1ns/1ps
module matrix_tile_group_loader_int4_ddr (
    input logic clk, input logic rst_n, input logic start,
    input logic [31:0] a_base, input logic [31:0] b_base,
    input logic [31:0] a_stride_bytes, input logic [31:0] b_stride_bytes,
    input logic [15:0] m0, input logic [15:0] n0, input logic [15:0] k0,
    input logic [3:0] m_len, input logic [3:0] k_len,
    input logic [4:0] n_cols, input logic [1:0] active_arrays,
    output logic busy, output logic done, output logic error,

    output logic read_req_valid, input logic read_req_ready,
    output logic [31:0] read_req_addr, output logic [4:0] read_req_len,
    input logic read_word_valid, output logic read_word_ready,
    input logic [31:0] read_word_data, input logic [3:0] read_word_keep,
    input logic read_word_last,

    output logic vector_valid, input logic vector_ready,
    output logic vector_array, output logic vector_matrix,
    output logic [2:0] vector_row, output logic [7:0] vector_mask,
    output logic [31:0] vector_data
);

    typedef enum logic {PHASE_A, PHASE_B} phase_t;

    logic running, issue_complete;
    phase_t issue_phase, receive_phase;
    logic [3:0] issue_row, receive_row;
    logic [31:0] issue_vector_addr, issue_word_addr;
    logic [1:0] issue_words_remaining;
    logic [31:0] a_first_addr_r, b_first_addr_r;
    logic [31:0] a_stride_r, b_stride_r;
    logic [3:0] a_rows_r, b_rows_r;
    logic [4:0] a_cols_r, b_cols_r;
    logic [1:0] arrays_r;

    logic [63:0] receive_partial, merged_data;
    logic [3:0] receive_gathered, receive_bytes, response_bytes;
    logic [1:0] receive_skip;
    logic [31:0] receive_vector_addr;
    logic response_completes;
    logic b1_pending, b1_last_pending;
    logic [2:0] b1_pending_row;
    logic [31:0] b1_pending_data;

    function automatic logic [3:0] packed_bytes(input logic [4:0] lanes);
        packed_bytes = ({1'b0, lanes} + 1'b1) >> 1;
    endfunction

    function automatic logic [7:0] active_mask(input logic [4:0] lanes);
        if (lanes >= 8)
            active_mask = 8'hff;
        else if (lanes == 0)
            active_mask = 8'h00;
        else
            active_mask = (9'h001 << lanes) - 1'b1;
    endfunction

    function automatic logic [1:0] word_count(
        input logic [1:0] byte_offset,
        input logic [3:0] byte_count
    );
        word_count = ({2'd0, byte_offset} + byte_count + 3) >> 2;
    endfunction

    function automatic logic [31:0] word_align(input logic [31:0] address);
        word_align = {address[31:2], 2'b00};
    endfunction

    function automatic logic [1:0] byte_offset(input logic [31:0] address);
        byte_offset = address[1:0];
    endfunction

    wire issue_fire = read_req_valid && read_req_ready;
    wire read_fire = read_word_valid && read_word_ready;
    wire vector_fire = vector_valid && vector_ready;
    wire issue_last_row = issue_phase == PHASE_A ?
                          issue_row + 1'b1 >= a_rows_r :
                          issue_row + 1'b1 >= b_rows_r;
    wire receive_last_row = receive_phase == PHASE_A ?
                            receive_row + 1'b1 >= a_rows_r :
                            receive_row + 1'b1 >= b_rows_r;
    wire [4:0] b1_cols = b_cols_r > 8 ? b_cols_r - 8 : 5'd0;

    assign busy = running;
    assign read_req_valid = running && !issue_complete;
    assign read_req_addr = issue_word_addr;
    assign read_req_len = 5'd4;

    always_comb begin
        receive_bytes = receive_phase == PHASE_A ?
                        packed_bytes(a_cols_r) : packed_bytes(b_cols_r);
        response_bytes = '0;
        merged_data = receive_partial;
        begin : merge_word
            integer destination;
            destination = receive_gathered;
            for (int lane = 0; lane < 4; lane++) begin
                if (read_word_keep[lane] && lane >= receive_skip &&
                    destination < receive_bytes) begin
                    merged_data[destination*8 +: 8] =
                        read_word_data[lane*8 +: 8];
                    destination = destination + 1;
                    response_bytes = response_bytes + 1'b1;
                end
            end
        end
        response_completes = read_word_valid &&
                             receive_gathered + response_bytes >=
                             receive_bytes;
    end

    assign vector_valid = b1_pending || response_completes;
    assign vector_array = b1_pending;
    assign vector_matrix = b1_pending || receive_phase == PHASE_B;
    assign vector_row = b1_pending ? b1_pending_row : receive_row[2:0];
    assign vector_mask = b1_pending ? active_mask(b1_cols) :
                         receive_phase == PHASE_A ?
                         active_mask(a_cols_r) :
                         active_mask(b_cols_r > 8 ? 5'd8 : b_cols_r);
    assign vector_data = b1_pending ? b1_pending_data :
                         merged_data[31:0];
    assign read_word_ready = b1_pending ? vector_ready :
                             (response_completes ? vector_ready : 1'b1);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            issue_complete <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            issue_phase <= PHASE_A;
            receive_phase <= PHASE_A;
            issue_row <= '0;
            receive_row <= '0;
            issue_vector_addr <= '0;
            issue_word_addr <= '0;
            issue_words_remaining <= '0;
            a_first_addr_r <= '0;
            b_first_addr_r <= '0;
            a_stride_r <= '0;
            b_stride_r <= '0;
            a_rows_r <= '0;
            b_rows_r <= '0;
            a_cols_r <= '0;
            b_cols_r <= '0;
            arrays_r <= '0;
            receive_partial <= '0;
            receive_gathered <= '0;
            receive_skip <= '0;
            receive_vector_addr <= '0;
            b1_pending <= 1'b0;
            b1_last_pending <= 1'b0;
            b1_pending_row <= '0;
            b1_pending_data <= '0;
        end else begin
            done <= 1'b0;

            if (start && !running) begin
                logic [31:0] first_a;
                logic [31:0] first_b;
                first_a = a_base + (m0 * a_stride_bytes) +
                          {17'd0, k0[15:1]};
                first_b = b_base + (k0 * b_stride_bytes) +
                          {17'd0, n0[15:1]};
                error <= 1'b0;
                issue_complete <= 1'b0;
                issue_phase <= PHASE_A;
                receive_phase <= PHASE_A;
                issue_row <= '0;
                receive_row <= '0;
                issue_vector_addr <= first_a;
                issue_word_addr <= word_align(first_a);
                issue_words_remaining <=
                    word_count(first_a[1:0], packed_bytes({1'b0, k_len}));
                a_first_addr_r <= first_a;
                b_first_addr_r <= first_b;
                a_stride_r <= a_stride_bytes;
                b_stride_r <= b_stride_bytes;
                a_rows_r <= m_len;
                b_rows_r <= k_len;
                a_cols_r <= {1'b0, k_len};
                b_cols_r <= n_cols;
                arrays_r <= active_arrays;
                receive_partial <= '0;
                receive_gathered <= '0;
                receive_skip <= first_a[1:0];
                receive_vector_addr <= first_a;
                b1_pending <= 1'b0;
                b1_last_pending <= 1'b0;
                if (m_len == 0 || k_len == 0 || n_cols == 0 ||
                    m_len > 8 || k_len > 8 || n_cols > 16 ||
                    active_arrays == 0 || active_arrays > 2 ||
                    k0[0] || n0[0]) begin
                    error <= 1'b1;
                    done <= 1'b1;
                end else begin
                    running <= 1'b1;
                end
            end

            if (issue_fire) begin
                if (issue_words_remaining > 1) begin
                    issue_word_addr <= issue_word_addr + 4;
                    issue_words_remaining <= issue_words_remaining - 1'b1;
                end else if (issue_phase == PHASE_A && issue_last_row) begin
                    issue_phase <= PHASE_B;
                    issue_row <= '0;
                    issue_vector_addr <= b_first_addr_r;
                    issue_word_addr <= word_align(b_first_addr_r);
                    issue_words_remaining <= word_count(
                        b_first_addr_r[1:0], packed_bytes(b_cols_r));
                end else if (issue_last_row) begin
                    issue_complete <= 1'b1;
                end else begin
                    logic [31:0] next_addr;
                    next_addr = issue_vector_addr +
                        (issue_phase == PHASE_A ? a_stride_r : b_stride_r);
                    issue_row <= issue_row + 1'b1;
                    issue_vector_addr <= next_addr;
                    issue_word_addr <= word_align(next_addr);
                    issue_words_remaining <= word_count(
                        next_addr[1:0],
                        issue_phase == PHASE_A ?
                        packed_bytes(a_cols_r) : packed_bytes(b_cols_r));
                end
            end

            if (read_fire) begin
                if (!read_word_last || response_bytes == 0) begin
                    error <= 1'b1;
                    running <= 1'b0;
                    done <= 1'b1;
                end else if (!response_completes) begin
                    receive_partial <= merged_data;
                    receive_gathered <= receive_gathered + response_bytes;
                    receive_skip <= '0;
                end else begin
                    receive_partial <= '0;
                    receive_gathered <= '0;
                    if (receive_phase == PHASE_A && receive_last_row) begin
                        receive_phase <= PHASE_B;
                        receive_row <= '0;
                        receive_vector_addr <= b_first_addr_r;
                        receive_skip <= b_first_addr_r[1:0];
                    end else if (receive_phase == PHASE_B &&
                                 arrays_r == 2) begin
                        b1_pending <= 1'b1;
                        b1_last_pending <= receive_last_row;
                        b1_pending_row <= receive_row[2:0];
                        b1_pending_data <= merged_data[63:32];
                        if (!receive_last_row) begin
                            receive_row <= receive_row + 1'b1;
                            receive_vector_addr <=
                                receive_vector_addr + b_stride_r;
                            receive_skip <= byte_offset(
                                receive_vector_addr + b_stride_r);
                        end
                    end else if (receive_last_row) begin
                        running <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        logic [31:0] next_receive_addr;
                        next_receive_addr = receive_vector_addr +
                            (receive_phase == PHASE_A ?
                             a_stride_r : b_stride_r);
                        receive_row <= receive_row + 1'b1;
                        receive_vector_addr <= next_receive_addr;
                        receive_skip <= next_receive_addr[1:0];
                    end
                end
            end

            if (b1_pending && vector_fire) begin
                b1_pending <= 1'b0;
                if (b1_last_pending) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end
                b1_last_pending <= 1'b0;
            end
        end
    end
endmodule
