`timescale 1ns/1ps
module gemm_dual_8x8_int4_core (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               load_valid,
    input  logic               load_array,
    input  logic               load_matrix,
    input  logic [5:0]         load_addr,
    input  logic signed [3:0]  load_data,
    input  logic               load_vector_valid,
    output logic               load_vector_ready,
    input  logic               load_vector_array,
    input  logic               load_vector_matrix,
    input  logic [2:0]         load_vector_row,
    input  logic [7:0]         load_vector_mask,
    input  logic [31:0]        load_vector_data,
    input  logic               load_group_valid,
    output logic               load_group_ready,
    input  logic [2:0]         load_group_row,
    input  logic [7:0]         load_group_a_mask,
    input  logic [31:0]        load_group_a_data,
    input  logic [7:0]         load_group_b0_mask,
    input  logic [31:0]        load_group_b0_data,
    input  logic [7:0]         load_group_b1_mask,
    input  logic [31:0]        load_group_b1_data,
    input  logic               start,
    input  logic [1:0]         share_mode,
    output logic               busy,
    output logic               done,
    input  logic               result_array,
    input  logic [5:0]         result_addr,
    output logic signed [31:0] result_data,
    input  logic [2:0]         result_row,
    output logic signed [255:0] result0_row,
    output logic signed [255:0] result1_row
);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_CLEAR,
        STATE_COMPUTE,
        STATE_DONE
    } state_t;

    state_t state;
    logic [4:0] compute_cycle;
    logic [1:0] share_mode_r;
    logic [31:0] a0_wave [0:14];
    logic [31:0] b0_wave [0:14];
    logic [31:0] a1_wave [0:14];
    logic [31:0] b1_wave [0:14];
    logic [31:0] a0_read;
    logic [31:0] b0_read;
    logic [31:0] a1_read;
    logic [31:0] b1_read;
    logic signed [31:0] a0_west_flat;
    logic signed [31:0] b0_north_flat;
    logic signed [31:0] a1_west_flat;
    logic signed [31:0] b1_north_flat;
    logic signed [32*64-1:0] c0_flat;
    logic signed [32*64-1:0] c1_flat;
    logic [3:0] load_lane;
    logic [3:0] load_wave_addr;
    logic vector_buffer_valid;
    logic vector_buffer_array;
    logic vector_buffer_matrix;
    logic [2:0] vector_buffer_row;
    logic [7:0] vector_buffer_mask;
    logic [31:0] vector_buffer_data;
    wire vector_commit = vector_buffer_valid && !busy;
    wire group_load_fire = load_group_valid && load_group_ready;

    assign load_wave_addr = {1'b0, load_addr[5:3]} + {1'b0, load_addr[2:0]};
    assign load_lane = load_matrix ? {1'b0, load_addr[2:0]} : {1'b0, load_addr[5:3]};
    assign busy = state == STATE_CLEAR || state == STATE_COMPUTE;
    assign load_group_ready = !busy && !vector_buffer_valid;
    assign load_vector_ready = !busy && !load_group_valid &&
                               (!vector_buffer_valid || vector_commit);
    assign result_data = result_array ?
                         c1_flat[result_addr*32 +: 32] :
                         c0_flat[result_addr*32 +: 32];
    assign result0_row = c0_flat[result_row*256 +: 256];
    assign result1_row = c1_flat[result_row*256 +: 256];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            vector_buffer_valid <= 1'b0;
            vector_buffer_array <= 1'b0;
            vector_buffer_matrix <= 1'b0;
            vector_buffer_row <= '0;
            vector_buffer_mask <= '0;
            vector_buffer_data <= '0;
        end else begin
            if (vector_commit)
                vector_buffer_valid <= 1'b0;
            if (load_vector_valid && load_vector_ready) begin
                vector_buffer_valid <= 1'b1;
                vector_buffer_array <= load_vector_array;
                vector_buffer_matrix <= load_vector_matrix;
                vector_buffer_row <= load_vector_row;
                vector_buffer_mask <= load_vector_mask;
                vector_buffer_data <= load_vector_data;
            end
        end

        if (group_load_fire) begin
            // Keep both dimensions of every wave write static. A variable
            // wave index combined with a variable nibble select makes Vivado
            // infer a large crossbar and can spend over an hour optimizing it.
            unique case (load_group_row)
                3'd0: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[lane][3:0] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd1: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[1 + lane][7:4] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[1 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[1 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd2: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[2 + lane][11:8] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[2 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[2 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd3: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[3 + lane][15:12] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[3 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[3 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd4: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[4 + lane][19:16] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[4 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[4 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd5: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[5 + lane][23:20] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[5 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[5 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd6: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[6 + lane][27:24] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[6 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[6 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                3'd7: begin
                    for (int lane = 0; lane < 8; lane++) begin
                        if (load_group_a_mask[lane])
                            a0_wave[7 + lane][31:28] <=
                                load_group_a_data[lane*4 +: 4];
                        if (load_group_b0_mask[lane])
                            b0_wave[7 + lane][lane*4 +: 4] <=
                                load_group_b0_data[lane*4 +: 4];
                        if (load_group_b1_mask[lane])
                            b1_wave[7 + lane][lane*4 +: 4] <=
                                load_group_b1_data[lane*4 +: 4];
                    end
                end
                default: begin
                end
            endcase
        end else if (vector_commit) begin
            for (int lane = 0; lane < 8; lane++) begin
                if (vector_buffer_mask[lane]) begin
                    case ({vector_buffer_array, vector_buffer_matrix})
                        2'b00: a0_wave[vector_buffer_row + lane]
                                      [vector_buffer_row*4 +: 4] <=
                                      vector_buffer_data[lane*4 +: 4];
                        2'b01: b0_wave[vector_buffer_row + lane]
                                      [lane*4 +: 4] <=
                                      vector_buffer_data[lane*4 +: 4];
                        2'b10: a1_wave[vector_buffer_row + lane]
                                      [vector_buffer_row*4 +: 4] <=
                                      vector_buffer_data[lane*4 +: 4];
                        2'b11: b1_wave[vector_buffer_row + lane]
                                      [lane*4 +: 4] <=
                                      vector_buffer_data[lane*4 +: 4];
                    endcase
                end
            end
        end else if (load_valid && !busy) begin
            case ({load_array, load_matrix})
                2'b00: a0_wave[load_wave_addr][load_lane*4 +: 4] <= load_data;
                2'b01: b0_wave[load_wave_addr][load_lane*4 +: 4] <= load_data;
                2'b10: a1_wave[load_wave_addr][load_lane*4 +: 4] <= load_data;
                2'b11: b1_wave[load_wave_addr][load_lane*4 +: 4] <= load_data;
            endcase
        end

        if (!rst_n) begin
            a0_read <= '0;
            b0_read <= '0;
            a1_read <= '0;
            b1_read <= '0;
        end else if (state == STATE_COMPUTE && compute_cycle <= 5'd14) begin
            a0_read <= a0_wave[compute_cycle];
            b0_read <= b0_wave[compute_cycle];
            a1_read <= a1_wave[compute_cycle];
            b1_read <= b1_wave[compute_cycle];
        end else begin
            a0_read <= '0;
            b0_read <= '0;
            a1_read <= '0;
            b1_read <= '0;
        end
    end

    always_comb begin
        a0_west_flat = '0;
        b0_north_flat = '0;
        a1_west_flat = '0;
        b1_north_flat = '0;

        if (state == STATE_COMPUTE) begin
            for (int row = 0; row < 8; row++) begin
                if (compute_cycle > row && compute_cycle - 1'b1 - row < 8) begin
                    a0_west_flat[row*4 +: 4] = a0_read[row*4 +: 4];
                    a1_west_flat[row*4 +: 4] = a1_read[row*4 +: 4];
                end
            end
            for (int col = 0; col < 8; col++) begin
                if (compute_cycle > col && compute_cycle - 1'b1 - col < 8) begin
                    b0_north_flat[col*4 +: 4] = b0_read[col*4 +: 4];
                    b1_north_flat[col*4 +: 4] = b1_read[col*4 +: 4];
                end
            end
        end
    end

    dual_systolic_array_8x8_int4 #(.USE_DSP(1'b0)) arrays (
        .clk(clk),
        .rst_n(rst_n),
        .clear(state == STATE_CLEAR),
        .enable(state == STATE_COMPUTE),
        .share_mode(share_mode_r),
        .a0_west_flat(a0_west_flat),
        .b0_north_flat(b0_north_flat),
        .a1_west_flat(a1_west_flat),
        .b1_north_flat(b1_north_flat),
        .c0_flat(c0_flat),
        .c1_flat(c1_flat)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            compute_cycle <= '0;
            share_mode_r <= '0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            unique case (state)
                STATE_IDLE: if (start) begin
                    share_mode_r <= share_mode;
                    compute_cycle <= '0;
                    state <= STATE_CLEAR;
                end

                STATE_CLEAR: state <= STATE_COMPUTE;

                STATE_COMPUTE: begin
                    if (compute_cycle == 5'd25) begin
                        compute_cycle <= '0;
                        state <= STATE_DONE;
                    end else begin
                        compute_cycle <= compute_cycle + 1'b1;
                    end
                end

                STATE_DONE: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
