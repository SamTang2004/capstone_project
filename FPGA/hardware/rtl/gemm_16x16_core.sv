module gemm_16x16_core (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        load_valid,
    input  logic        load_matrix,      // 0 = A, 1 = B
    input  logic [7:0]  load_addr,
    input  logic [7:0]  load_data,

    input  logic        start,
    output logic        busy,
    output logic        done,

    input  logic [7:0]  result_addr,
    output logic [31:0] result_data
);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_CLEAR,
        STATE_COMPUTE,
        STATE_DONE
    } state_t;

    state_t state;

    // The array consumes one A value per row and one B value per column on
    // every cycle.  Bank by row/column so each value has its own synchronous
    // memory read port instead of relying on a 16-read combinational array.
    // Each systolic lane needs an independent synchronous read every cycle.
    // Give every lane a BRAM-sized bank rather than a 16-byte LUTRAM.  The
    // current controller uses addresses 0..15; the remaining depth reserves
    // room for future ping-pong tile prefetch without changing this interface.
    (* ram_style = "block", ram_decomp = "power" *) logic signed [7:0] a_bank [0:15][0:255];
    (* ram_style = "block", ram_decomp = "power" *) logic signed [7:0] b_bank [0:15][0:255];
    logic signed [7:0] a_read_data [0:15];
    logic signed [7:0] b_read_data [0:15];

    logic [7:0] clear_idx;
    logic [5:0] compute_cycle;

    logic signed [8*16-1:0]   a_west_flat;
    logic signed [8*16-1:0]   b_north_flat;
    logic signed [32*256-1:0] c_flat;

    assign result_data = c_flat[result_addr*32 +: 32];

    systolic_array_16x16 array (
        .clk(clk),
        .rst_n(rst_n),
        .clear(state == STATE_CLEAR),
        .enable(state == STATE_COMPUTE),
        .a_west_flat(a_west_flat),
        .b_north_flat(b_north_flat),
        .c_flat(c_flat)
    );

    always_comb begin
        a_west_flat = '0;
        b_north_flat = '0;

        if (state == STATE_COMPUTE) begin
            for (int r = 0; r < 16; r++) begin
                a_west_flat[r*8 +: 8] = a_read_data[r];
            end

            for (int c = 0; c < 16; c++) begin
                b_north_flat[c*8 +: 8] = b_read_data[c];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            clear_idx <= '0;
            compute_cycle <= '0;
        end else begin
            done <= 1'b0;

            if (load_valid && !busy) begin
                if (load_matrix) begin
                    b_bank[load_addr[3:0]][load_addr[7:4]] <= load_data;
                end else begin
                    a_bank[load_addr[7:4]][load_addr[3:0]] <= load_data;
                end
            end

            // Registered tile-memory reads form the one-cycle memory stage.
            // Invalid wavefront positions inject zeroes for array flushing.
            if (state == STATE_COMPUTE) begin
                for (int r = 0; r < 16; r++) begin
                    if (compute_cycle >= r && compute_cycle - r < 16) begin
                        a_read_data[r] <= a_bank[r][compute_cycle - r];
                    end else begin
                        a_read_data[r] <= '0;
                    end
                end
                for (int c = 0; c < 16; c++) begin
                    if (compute_cycle >= c && compute_cycle - c < 16) begin
                        b_read_data[c] <= b_bank[c][compute_cycle - c];
                    end else begin
                        b_read_data[c] <= '0;
                    end
                end
            end else begin
                for (int lane = 0; lane < 16; lane++) begin
                    a_read_data[lane] <= '0;
                    b_read_data[lane] <= '0;
                end
            end

            unique case (state)
                STATE_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        clear_idx <= '0;
                        state <= STATE_CLEAR;
                    end
                end

                STATE_CLEAR: begin
                    clear_idx <= clear_idx + 8'd1;
                    if (clear_idx == 8'd0) begin
                        compute_cycle <= '0;
                        state <= STATE_COMPUTE;
                    end
                end

                STATE_COMPUTE: begin
                    // One extra cycle drains the registered tile-memory
                    // stage before the systolic wavefront has fully exited.
                    if (compute_cycle == 6'd47) begin
                        compute_cycle <= '0;
                        state <= STATE_DONE;
                    end else begin
                        compute_cycle <= compute_cycle + 6'd1;
                    end
                end

                STATE_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule
