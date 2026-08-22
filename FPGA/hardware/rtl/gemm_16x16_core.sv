module gemm_16x16_core (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        load_valid,
    input  logic        load_matrix,      // 0 = A, 1 = B
    input  logic [7:0]  load_addr,
    input  logic [7:0]  load_data,
    input  logic        load_word_valid,
    input  logic [7:0]  load_word_addr,
    input  logic [7:0]  load_word_stride,
    input  logic [63:0] load_word_data,
    input  logic [7:0]  load_word_keep,

    input  logic        invalidate_valid,
    input  logic        invalidate_matrix,

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

    logic load_valid_r, load_matrix_r, load_word_valid_r;
    logic [7:0] load_addr_r, load_data_r;
    logic [7:0] load_word_addr_r, load_word_stride_r;
    logic [63:0] load_word_data_r;
    logic [7:0] load_word_keep_r;

    // Bank by wave address modulo eight. Eight consecutive tile bytes reach
    // independent BRAM write ports, matching one 64-bit HP0 response beat.
    logic [127:0] a_wave_bank_din  [0:7];
    logic [127:0] a_wave_bank_dout [0:7];
    logic [127:0] b_wave_bank_din  [0:7];
    logic [127:0] b_wave_bank_dout [0:7];
    logic [15:0]  a_wave_bank_we   [0:7];
    logic [15:0]  b_wave_bank_we   [0:7];
    logic [127:0] a_wave_read_data;
    logic [127:0] b_wave_read_data;
    logic [2:0] wave_read_bank;
    logic [5:0] load_wave_addr;
    logic [3:0] load_wave_lane;
    logic [7:0] bank_we;
    logic [3:0] bank_addr [0:7];
    logic [3:0] bank_lane [0:7];
    logic [7:0] bank_data [0:7];
    logic       bank_write_valid_r;
    logic       bank_write_matrix_r;
    logic [7:0] bank_we_r;
    logic [3:0] bank_addr_r [0:7];
    logic [3:0] bank_lane_r [0:7];
    logic [7:0] bank_data_r [0:7];
    logic [15:0] a_valid_bank [0:7][0:3];
    logic [15:0] b_valid_bank [0:7][0:3];
    logic [15:0] a_wave_valid_data;
    logic [15:0] b_wave_valid_data;

    logic [7:0] clear_idx;
    logic [5:0] compute_cycle;

    logic signed [8*16-1:0]   a_west_flat;
    logic signed [8*16-1:0]   b_north_flat;
    logic signed [8*16-1:0]   a_west_pipe;
    logic signed [8*16-1:0]   b_north_pipe;
    logic signed [32*256-1:0] c_flat;

    assign result_data = c_flat[result_addr*32 +: 32];
    assign load_wave_addr = {2'b0, load_addr_r[7:4]} + {2'b0, load_addr_r[3:0]};
    assign load_wave_lane = load_matrix_r ? load_addr_r[3:0] : load_addr_r[7:4];

    // Decouple the loader's address/stride registers from the replicated BRAM
    // write decode. This stage accepts one 64-bit beat every clock.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            load_valid_r <= 1'b0;
            load_matrix_r <= 1'b0;
            load_addr_r <= '0;
            load_data_r <= '0;
            load_word_valid_r <= 1'b0;
            load_word_addr_r <= '0;
            load_word_stride_r <= '0;
            load_word_data_r <= '0;
            load_word_keep_r <= '0;
        end else begin
            load_valid_r <= load_valid;
            load_matrix_r <= load_matrix;
            load_addr_r <= load_addr;
            load_data_r <= load_data;
            load_word_valid_r <= load_word_valid;
            load_word_addr_r <= load_word_addr;
            load_word_stride_r <= load_word_stride;
            load_word_data_r <= load_word_data;
            load_word_keep_r <= load_word_keep;
        end
    end

    always_comb begin
        logic [7:0] lane_tile_addr;
        logic [5:0] lane_wave_addr;
        logic [3:0] lane_wave_lane;
        bank_we = '0;
        for (int bank = 0; bank < 8; bank++) begin
            bank_addr[bank] = '0;
            bank_lane[bank] = '0;
            bank_data[bank] = '0;
        end
        if (load_word_valid_r) begin
            for (int lane = 0; lane < 8; lane++) begin
                lane_tile_addr = load_word_addr_r + lane * load_word_stride_r;
                lane_wave_addr = {2'b0, lane_tile_addr[7:4]} + {2'b0, lane_tile_addr[3:0]};
                lane_wave_lane = load_matrix_r ? lane_tile_addr[3:0] : lane_tile_addr[7:4];
                if (load_word_keep_r[lane]) begin
                    bank_we[lane_wave_addr[2:0]] = 1'b1;
                    bank_addr[lane_wave_addr[2:0]] = {1'b0, lane_wave_addr[5:3]};
                    bank_lane[lane_wave_addr[2:0]] = lane_wave_lane;
                    bank_data[lane_wave_addr[2:0]] = load_word_data_r[lane*8 +: 8];
                end
            end
        end else if (load_valid_r) begin
            bank_we[load_wave_addr[2:0]] = 1'b1;
            bank_addr[load_wave_addr[2:0]] = {1'b0, load_wave_addr[5:3]};
            bank_lane[load_wave_addr[2:0]] = load_wave_lane;
            bank_data[load_wave_addr[2:0]] = load_data_r;
        end
    end

    // Register the complete lane-to-bank decode. The stride/address arithmetic
    // is replicated across eight lanes, so it must not feed BRAM data and write
    // enables in the same cycle. This stage remains fully pipelined.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            bank_write_valid_r <= 1'b0;
            bank_write_matrix_r <= 1'b0;
            bank_we_r <= '0;
            for (int bank = 0; bank < 8; bank++) begin
                bank_addr_r[bank] <= '0;
                bank_lane_r[bank] <= '0;
                bank_data_r[bank] <= '0;
            end
        end else begin
            bank_write_valid_r <= (load_valid_r || load_word_valid_r) && !busy;
            bank_write_matrix_r <= load_matrix_r;
            bank_we_r <= bank_we;
            for (int bank = 0; bank < 8; bank++) begin
                bank_addr_r[bank] <= bank_addr[bank];
                bank_lane_r[bank] <= bank_lane[bank];
                bank_data_r[bank] <= bank_data[bank];
            end
        end
    end

    always_comb begin
        for (int bank = 0; bank < 8; bank++) begin
            a_wave_bank_din[bank] = '0;
            b_wave_bank_din[bank] = '0;
            a_wave_bank_we[bank] = '0;
            b_wave_bank_we[bank] = '0;
            if (bank_write_valid_r && bank_we_r[bank]) begin
                if (!bank_write_matrix_r) begin
                    a_wave_bank_din[bank][bank_lane_r[bank]*8 +: 8] = bank_data_r[bank];
                    a_wave_bank_we[bank][bank_lane_r[bank]] = 1'b1;
                end else begin
                    b_wave_bank_din[bank][bank_lane_r[bank]*8 +: 8] = bank_data_r[bank];
                    b_wave_bank_we[bank][bank_lane_r[bank]] = 1'b1;
                end
            end
        end
    end

    for (genvar wave_bank = 0; wave_bank < 8; wave_bank++) begin : gen_wave_bram
        xpm_memory_sdpram #(
            .MEMORY_SIZE(512),
            .MEMORY_PRIMITIVE("block"),
            .CLOCKING_MODE("common_clock"),
            .MEMORY_INIT_FILE("none"),
            .USE_MEM_INIT(0),
            .WRITE_DATA_WIDTH_A(128),
            .BYTE_WRITE_WIDTH_A(8),
            .ADDR_WIDTH_A(2),
            .READ_DATA_WIDTH_B(128),
            .ADDR_WIDTH_B(2),
            .READ_RESET_VALUE_B("0"),
            .READ_LATENCY_B(1),
            .WRITE_MODE_B("no_change"),
            .RST_MODE_B("SYNC")
        ) a_wave_memory (
            .sleep(1'b0),
            .clka(clk),
            .ena(|a_wave_bank_we[wave_bank]),
            .wea(a_wave_bank_we[wave_bank]),
            .addra(bank_addr_r[wave_bank][1:0]),
            .dina(a_wave_bank_din[wave_bank]),
            .injectsbiterra(1'b0),
            .injectdbiterra(1'b0),
            .clkb(clk),
            .rstb(!rst_n),
            .enb(state == STATE_COMPUTE && compute_cycle <= 6'd30),
            .regceb(1'b1),
            .addrb(compute_cycle[4:3]),
            .doutb(a_wave_bank_dout[wave_bank]),
            .sbiterrb(),
            .dbiterrb()
        );

        xpm_memory_sdpram #(
            .MEMORY_SIZE(512),
            .MEMORY_PRIMITIVE("block"),
            .CLOCKING_MODE("common_clock"),
            .MEMORY_INIT_FILE("none"),
            .USE_MEM_INIT(0),
            .WRITE_DATA_WIDTH_A(128),
            .BYTE_WRITE_WIDTH_A(8),
            .ADDR_WIDTH_A(2),
            .READ_DATA_WIDTH_B(128),
            .ADDR_WIDTH_B(2),
            .READ_RESET_VALUE_B("0"),
            .READ_LATENCY_B(1),
            .WRITE_MODE_B("no_change"),
            .RST_MODE_B("SYNC")
        ) b_wave_memory (
            .sleep(1'b0),
            .clka(clk),
            .ena(|b_wave_bank_we[wave_bank]),
            .wea(b_wave_bank_we[wave_bank]),
            .addra(bank_addr_r[wave_bank][1:0]),
            .dina(b_wave_bank_din[wave_bank]),
            .injectsbiterra(1'b0),
            .injectdbiterra(1'b0),
            .clkb(clk),
            .rstb(!rst_n),
            .enb(state == STATE_COMPUTE && compute_cycle <= 6'd30),
            .regceb(1'b1),
            .addrb(compute_cycle[4:3]),
            .doutb(b_wave_bank_dout[wave_bank]),
            .sbiterrb(),
            .dbiterrb()
        );
    end

    always_comb begin
        a_wave_read_data = a_wave_bank_dout[wave_read_bank];
        b_wave_read_data = b_wave_bank_dout[wave_read_bank];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wave_read_bank <= '0;
            a_wave_valid_data <= '0;
            b_wave_valid_data <= '0;
        end else if (state == STATE_COMPUTE && compute_cycle <= 6'd30) begin
            wave_read_bank <= compute_cycle[2:0];
            a_wave_valid_data <= a_valid_bank[compute_cycle[2:0]][compute_cycle[4:3]];
            b_wave_valid_data <= b_valid_bank[compute_cycle[2:0]][compute_cycle[4:3]];
        end
    end

    // Partial edge tiles invalidate in one clock. Unwritten wave lanes are
    // forced to zero at the array boundary instead of scrubbing every BRAM byte.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int bank = 0; bank < 8; bank++) begin
                for (int address = 0; address < 4; address++) begin
                    a_valid_bank[bank][address] <= '0;
                    b_valid_bank[bank][address] <= '0;
                end
            end
        end else begin
            if (invalidate_valid) begin
                for (int bank = 0; bank < 8; bank++) begin
                    for (int address = 0; address < 4; address++) begin
                        if (invalidate_matrix) b_valid_bank[bank][address] <= '0;
                        else a_valid_bank[bank][address] <= '0;
                    end
                end
            end
            if (bank_write_valid_r) begin
                for (int bank = 0; bank < 8; bank++) begin
                    if (bank_we_r[bank]) begin
                        if (bank_write_matrix_r)
                            b_valid_bank[bank][bank_addr_r[bank]][bank_lane_r[bank]] <= 1'b1;
                        else
                            a_valid_bank[bank][bank_addr_r[bank]][bank_lane_r[bank]] <= 1'b1;
                    end
                end
            end
        end
    end

    systolic_array_16x16 array (
        .clk(clk),
        .rst_n(rst_n),
        .clear(state == STATE_CLEAR),
        .enable(state == STATE_COMPUTE),
        .a_west_flat(a_west_pipe),
        .b_north_flat(b_north_pipe),
        .c_flat(c_flat)
    );

    always_comb begin
        a_west_flat = '0;
        b_north_flat = '0;

        if (state == STATE_COMPUTE) begin
            for (int r = 0; r < 16; r++) begin
                if (compute_cycle > r && compute_cycle - 1'b1 - r < 16 && a_wave_valid_data[r])
                    a_west_flat[r*8 +: 8] = a_wave_read_data[r*8 +: 8];
            end

            for (int c = 0; c < 16; c++) begin
                if (compute_cycle > c && compute_cycle - 1'b1 - c < 16 && b_wave_valid_data[c])
                    b_north_flat[c*8 +: 8] = b_wave_read_data[c*8 +: 8];
            end
        end
    end

    // Register the fully selected wavefront before the PE boundary. This
    // isolates BRAM output muxing and valid-mask logic from the array routing.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_west_pipe <= '0;
            b_north_pipe <= '0;
        end else begin
            a_west_pipe <= a_west_flat;
            b_north_pipe <= b_north_flat;
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
                    // Drain the registered tile-memory, PE operand, and PE
                    // multiplier stages after the systolic wavefront exits.
                    if (compute_cycle == 6'd49) begin
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
