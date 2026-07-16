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

    // Four 32-bit BRAM slices form each 128-bit pre-skewed wave memory.
    // Static byte-write cases preserve one-byte loading without creating a
    // variable packed-memory selector in simulation or synthesis.
    (* ram_style = "block" *) logic [31:0] a_wave0 [0:63], a_wave1 [0:63], a_wave2 [0:63], a_wave3 [0:63];
    (* ram_style = "block" *) logic [31:0] b_wave0 [0:63], b_wave1 [0:63], b_wave2 [0:63], b_wave3 [0:63];
    logic [127:0] a_wave_read_data;
    logic [127:0] b_wave_read_data;
    logic [5:0] load_wave_addr;
    logic [3:0] load_wave_lane;

    logic [7:0] clear_idx;
    logic [5:0] compute_cycle;

    logic signed [8*16-1:0]   a_west_flat;
    logic signed [8*16-1:0]   b_north_flat;
    logic signed [32*256-1:0] c_flat;

    assign result_data = c_flat[result_addr*32 +: 32];
    assign load_wave_addr = {2'b0, load_addr[7:4]} + {2'b0, load_addr[3:0]};
    assign load_wave_lane = load_matrix ? load_addr[3:0] : load_addr[7:4];

    always_ff @(posedge clk) begin
        if (load_valid && !busy) begin
            if (!load_matrix) begin
                case (load_wave_lane)
                    4'd0: a_wave0[load_wave_addr][7:0] <= load_data; 4'd1: a_wave0[load_wave_addr][15:8] <= load_data;
                    4'd2: a_wave0[load_wave_addr][23:16] <= load_data; 4'd3: a_wave0[load_wave_addr][31:24] <= load_data;
                    4'd4: a_wave1[load_wave_addr][7:0] <= load_data; 4'd5: a_wave1[load_wave_addr][15:8] <= load_data;
                    4'd6: a_wave1[load_wave_addr][23:16] <= load_data; 4'd7: a_wave1[load_wave_addr][31:24] <= load_data;
                    4'd8: a_wave2[load_wave_addr][7:0] <= load_data; 4'd9: a_wave2[load_wave_addr][15:8] <= load_data;
                    4'd10: a_wave2[load_wave_addr][23:16] <= load_data; 4'd11: a_wave2[load_wave_addr][31:24] <= load_data;
                    4'd12: a_wave3[load_wave_addr][7:0] <= load_data; 4'd13: a_wave3[load_wave_addr][15:8] <= load_data;
                    4'd14: a_wave3[load_wave_addr][23:16] <= load_data; default: a_wave3[load_wave_addr][31:24] <= load_data;
                endcase
            end else begin
                case (load_wave_lane)
                    4'd0: b_wave0[load_wave_addr][7:0] <= load_data; 4'd1: b_wave0[load_wave_addr][15:8] <= load_data;
                    4'd2: b_wave0[load_wave_addr][23:16] <= load_data; 4'd3: b_wave0[load_wave_addr][31:24] <= load_data;
                    4'd4: b_wave1[load_wave_addr][7:0] <= load_data; 4'd5: b_wave1[load_wave_addr][15:8] <= load_data;
                    4'd6: b_wave1[load_wave_addr][23:16] <= load_data; 4'd7: b_wave1[load_wave_addr][31:24] <= load_data;
                    4'd8: b_wave2[load_wave_addr][7:0] <= load_data; 4'd9: b_wave2[load_wave_addr][15:8] <= load_data;
                    4'd10: b_wave2[load_wave_addr][23:16] <= load_data; 4'd11: b_wave2[load_wave_addr][31:24] <= load_data;
                    4'd12: b_wave3[load_wave_addr][7:0] <= load_data; 4'd13: b_wave3[load_wave_addr][15:8] <= load_data;
                    4'd14: b_wave3[load_wave_addr][23:16] <= load_data; default: b_wave3[load_wave_addr][31:24] <= load_data;
                endcase
            end
        end

        if (!rst_n) begin
            a_wave_read_data <= '0; b_wave_read_data <= '0;
        end else if (state == STATE_COMPUTE && compute_cycle <= 6'd30) begin
            a_wave_read_data <= {a_wave3[compute_cycle], a_wave2[compute_cycle], a_wave1[compute_cycle], a_wave0[compute_cycle]};
            b_wave_read_data <= {b_wave3[compute_cycle], b_wave2[compute_cycle], b_wave1[compute_cycle], b_wave0[compute_cycle]};
        end else begin
            a_wave_read_data <= '0; b_wave_read_data <= '0;
        end
    end

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
                if (compute_cycle > r && compute_cycle - 1'b1 - r < 16)
                    a_west_flat[r*8 +: 8] = a_wave_read_data[r*8 +: 8];
            end

            for (int c = 0; c < 16; c++) begin
                if (compute_cycle > c && compute_cycle - 1'b1 - c < 16)
                    b_north_flat[c*8 +: 8] = b_wave_read_data[c*8 +: 8];
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
