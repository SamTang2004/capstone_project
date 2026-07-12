module gemm_controller #(
    parameter int DIM_WIDTH = 16,
    parameter int MAX_TILE_SIZE = 16,
    parameter int MAX_ARRAYS = 8
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       start,
    input  logic [DIM_WIDTH-1:0]       M,
    input  logic [DIM_WIDTH-1:0]       N,
    input  logic [DIM_WIDTH-1:0]       K,
    input  logic [4:0]                 tile_size,
    input  logic [7:0]                 array_count,

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

    output logic [DIM_WIDTH-1:0]       m_tile,
    output logic [DIM_WIDTH-1:0]       n_group_tile,
    output logic [DIM_WIDTH-1:0]       k_tile,
    output logic [DIM_WIDTH-1:0]       m0,
    output logic [DIM_WIDTH-1:0]       n0,
    output logic [DIM_WIDTH-1:0]       k0,
    output logic [4:0]                 m_len,
    output logic [DIM_WIDTH-1:0]       n_group_cols,
    output logic [4:0]                 k_len,
    output logic [7:0]                 active_arrays,
    output logic                       clear_accumulators,
    output logic                       write_c_tiles
);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_LOAD_A_START,
        STATE_LOAD_A_WAIT,
        STATE_LOAD_B_START,
        STATE_LOAD_B_WAIT,
        STATE_COMPUTE_START,
        STATE_COMPUTE_WAIT,
        STATE_WRITE_C_START,
        STATE_WRITE_C_WAIT,
        STATE_ADVANCE,
        STATE_DONE,
        STATE_ERROR
    } state_t;

    state_t state;

    logic [DIM_WIDTH-1:0] M_latched;
    logic [DIM_WIDTH-1:0] N_latched;
    logic [DIM_WIDTH-1:0] K_latched;
    logic [4:0]           tile_size_latched;
    logic [7:0]           array_count_latched;
    logic [DIM_WIDTH-1:0] m_tiles;
    logic [DIM_WIDTH-1:0] n_tiles;
    logic [DIM_WIDTH-1:0] k_tiles;
    logic [DIM_WIDTH-1:0] m_tile_r;
    logic [DIM_WIDTH-1:0] n_group_tile_r;
    logic [DIM_WIDTH-1:0] k_tile_r;
    logic [DIM_WIDTH-1:0] active_arrays_dim;

    function automatic logic [DIM_WIDTH-1:0] ceil_div_tile(input logic [DIM_WIDTH-1:0] value);
        logic [DIM_WIDTH:0] adjusted;
        begin
            adjusted = value + (MAX_TILE_SIZE - 1);
            ceil_div_tile = adjusted >> 4;
        end
    endfunction

    function automatic logic [4:0] tile_extent(
        input logic [DIM_WIDTH-1:0] dim,
        input logic [DIM_WIDTH-1:0] offset,
        input logic [4:0] tile
    );
        logic [DIM_WIDTH-1:0] remaining;
        begin
            remaining = dim - offset;
            tile_extent = (remaining > tile) ? tile : remaining[4:0];
        end
    endfunction

    function automatic logic [DIM_WIDTH-1:0] min_dim(
        input logic [DIM_WIDTH-1:0] a,
        input logic [DIM_WIDTH-1:0] b
    );
        begin
            min_dim = (a < b) ? a : b;
        end
    endfunction

    assign busy = (state != STATE_IDLE && state != STATE_DONE && state != STATE_ERROR);
    assign load_a_start = (state == STATE_LOAD_A_START);
    assign load_b_start = (state == STATE_LOAD_B_START);
    assign compute_start = (state == STATE_COMPUTE_START);
    assign write_c_start = (state == STATE_WRITE_C_START);

    assign m_tile = m_tile_r;
    assign n_group_tile = n_group_tile_r;
    assign k_tile = k_tile_r;
    assign m0 = m_tile_r << 4;
    assign n0 = n_group_tile_r << 4;
    assign k0 = k_tile_r << 4;
    assign m_len = tile_extent(M_latched, m0, MAX_TILE_SIZE);
    assign k_len = tile_extent(K_latched, k0, MAX_TILE_SIZE);
    assign active_arrays_dim = min_dim(array_count_latched, n_tiles - n_group_tile_r);
    assign active_arrays = active_arrays_dim[7:0];
    assign n_group_cols = min_dim(N_latched - n0, active_arrays_dim << 4);
    assign clear_accumulators = busy && (k_tile_r == '0);
    assign write_c_tiles = busy && (k_tile_r == k_tiles - 1'b1);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            done <= 1'b0;
            error <= 1'b0;
            M_latched <= '0;
            N_latched <= '0;
            K_latched <= '0;
            tile_size_latched <= '0;
            array_count_latched <= '0;
            m_tiles <= '0;
            n_tiles <= '0;
            k_tiles <= '0;
            m_tile_r <= '0;
            n_group_tile_r <= '0;
            k_tile_r <= '0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: begin
                    error <= 1'b0;
                    if (start) begin
                        if (M == 0 || N == 0 || K == 0 || tile_size != MAX_TILE_SIZE || array_count == 0 || array_count > MAX_ARRAYS) begin
                            error <= 1'b1;
                            state <= STATE_ERROR;
                        end else begin
                            M_latched <= M;
                            N_latched <= N;
                            K_latched <= K;
                            tile_size_latched <= tile_size;
                            array_count_latched <= array_count;
                            m_tiles <= ceil_div_tile(M);
                            n_tiles <= ceil_div_tile(N);
                            k_tiles <= ceil_div_tile(K);
                            m_tile_r <= '0;
                            n_group_tile_r <= '0;
                            k_tile_r <= '0;
                            state <= STATE_LOAD_A_START;
                        end
                    end
                end

                STATE_LOAD_A_START: state <= STATE_LOAD_A_WAIT;
                STATE_LOAD_A_WAIT: begin
                    if (load_a_done) begin
                        state <= STATE_LOAD_B_START;
                    end
                end

                STATE_LOAD_B_START: state <= STATE_LOAD_B_WAIT;
                STATE_LOAD_B_WAIT: begin
                    if (load_b_done) begin
                        state <= STATE_COMPUTE_START;
                    end
                end

                STATE_COMPUTE_START: state <= STATE_COMPUTE_WAIT;
                STATE_COMPUTE_WAIT: begin
                    if (compute_done) begin
                        state <= (k_tile_r == k_tiles - 1'b1) ? STATE_WRITE_C_START : STATE_ADVANCE;
                    end
                end

                STATE_WRITE_C_START: state <= STATE_WRITE_C_WAIT;
                STATE_WRITE_C_WAIT: begin
                    if (write_c_done) begin
                        state <= STATE_ADVANCE;
                    end
                end

                STATE_ADVANCE: begin
                    if (k_tile_r + 1'b1 < k_tiles) begin
                        k_tile_r <= k_tile_r + 1'b1;
                        state <= STATE_LOAD_A_START;
                    end else begin
                        k_tile_r <= '0;
                        if (n_group_tile_r + array_count_latched < n_tiles) begin
                            n_group_tile_r <= n_group_tile_r + array_count_latched;
                            state <= STATE_LOAD_A_START;
                        end else begin
                            n_group_tile_r <= '0;
                            if (m_tile_r + 1'b1 < m_tiles) begin
                                m_tile_r <= m_tile_r + 1'b1;
                                state <= STATE_LOAD_A_START;
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
