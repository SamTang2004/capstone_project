`timescale 1ns/1ps
module gemm_controller_int4 #(
    parameter int DIM_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       start,
    input  logic [DIM_WIDTH-1:0]       M,
    input  logic [DIM_WIDTH-1:0]       N,
    input  logic [DIM_WIDTH-1:0]       K,

    output logic                       busy,
    output logic                       done,
    output logic                       error,
    output logic                       load_a_start,
    input  logic                       load_a_done,
    output logic                       load_b_start,
    input  logic                       load_b_done,
    output logic                       compute_start,
    input  logic                       compute_done,
    output logic                       write_c_start,
    input  logic                       write_c_done,

    output logic [DIM_WIDTH-1:0]       m0,
    output logic [DIM_WIDTH-1:0]       n0,
    output logic [DIM_WIDTH-1:0]       k0,
    output logic [3:0]                 m_len,
    output logic [3:0]                 k_len,
    output logic [4:0]                 n_group_cols,
    output logic [1:0]                 active_arrays,
    output logic                       clear_accumulators,
    output logic                       write_c_tiles
);

    localparam logic [DIM_WIDTH-1:0] TILE_SIZE = 8;
    localparam logic [DIM_WIDTH-1:0] GROUP_SIZE = 16;

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_PREPARE,
        STATE_LOAD_A_START,
        STATE_LOAD_A_WAIT,
        STATE_LOAD_B_START,
        STATE_LOAD_B_WAIT,
        STATE_COMPUTE_START,
        STATE_COMPUTE_WAIT,
        STATE_WRITE_START,
        STATE_WRITE_WAIT,
        STATE_ADVANCE,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;
    logic [DIM_WIDTH-1:0] m_dim;
    logic [DIM_WIDTH-1:0] n_dim;
    logic [DIM_WIDTH-1:0] k_dim;
    logic [DIM_WIDTH-1:0] m_pos;
    logic [DIM_WIDTH-1:0] n_pos;
    logic [DIM_WIDTH-1:0] k_pos;
    logic [DIM_WIDTH-1:0] m_remaining;
    logic [DIM_WIDTH-1:0] n_remaining;
    logic [DIM_WIDTH-1:0] k_remaining;

    assign busy = state != STATE_IDLE && state != STATE_DONE && state != STATE_ERROR;
    assign load_a_start = state == STATE_LOAD_A_START;
    assign load_b_start = state == STATE_LOAD_B_START;
    assign compute_start = state == STATE_COMPUTE_START;
    assign write_c_start = state == STATE_WRITE_START;

    assign m0 = m_pos;
    assign n0 = n_pos;
    assign k0 = k_pos;
    assign m_remaining = m_dim - m_pos;
    assign n_remaining = n_dim - n_pos;
    assign k_remaining = k_dim - k_pos;
    assign m_len = (m_remaining > TILE_SIZE) ? 4'd8 : m_remaining[3:0];
    assign k_len = (k_remaining > TILE_SIZE) ? 4'd8 : k_remaining[3:0];
    assign n_group_cols = (n_remaining > GROUP_SIZE) ? 5'd16 : n_remaining[4:0];
    assign active_arrays = (n_remaining > TILE_SIZE) ? 2'd2 : 2'd1;
    assign clear_accumulators = busy && k_pos == 0;
    assign write_c_tiles = busy && k_remaining <= TILE_SIZE;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            m_dim <= '0;
            n_dim <= '0;
            k_dim <= '0;
            m_pos <= '0;
            n_pos <= '0;
            k_pos <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        if (M == 0 || N == 0 || K == 0) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            m_dim <= M;
                            n_dim <= N;
                            k_dim <= K;
                            m_pos <= '0;
                            n_pos <= '0;
                            k_pos <= '0;
                            state <= STATE_PREPARE;
                        end
                    end
                end

                STATE_PREPARE: state <= STATE_LOAD_A_START;
                STATE_LOAD_A_START: state <= STATE_LOAD_A_WAIT;
                STATE_LOAD_A_WAIT: if (load_a_done) state <= STATE_LOAD_B_START;
                STATE_LOAD_B_START: state <= STATE_LOAD_B_WAIT;
                STATE_LOAD_B_WAIT: if (load_b_done) state <= STATE_COMPUTE_START;
                STATE_COMPUTE_START: state <= STATE_COMPUTE_WAIT;
                STATE_COMPUTE_WAIT: if (compute_done) begin
                    state <= (k_remaining <= TILE_SIZE) ? STATE_WRITE_START : STATE_ADVANCE;
                end
                STATE_WRITE_START: state <= STATE_WRITE_WAIT;
                STATE_WRITE_WAIT: if (write_c_done) state <= STATE_ADVANCE;

                STATE_ADVANCE: begin
                    if (k_remaining > TILE_SIZE) begin
                        k_pos <= k_pos + TILE_SIZE;
                        state <= STATE_LOAD_A_START;
                    end else begin
                        k_pos <= '0;
                        if (n_remaining > GROUP_SIZE) begin
                            n_pos <= n_pos + GROUP_SIZE;
                            state <= STATE_PREPARE;
                        end else begin
                            n_pos <= '0;
                            if (m_remaining > TILE_SIZE) begin
                                m_pos <= m_pos + TILE_SIZE;
                                state <= STATE_PREPARE;
                            end else begin
                                state <= STATE_DONE;
                            end
                        end
                    end
                end

                STATE_DONE: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                STATE_ERROR: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
