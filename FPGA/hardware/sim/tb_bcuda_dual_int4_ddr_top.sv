`timescale 1ns/1ps
module tb_bcuda_dual_int4_ddr_top;
    localparam int MEM_BYTES = 4096;
    localparam int A_BASE = 32'h100;
    localparam int B_BASE = 32'h400;
    localparam int C_BASE = 32'h800;
    localparam int M = 5;
    localparam int N = 11;
    localparam int K = 25;
    localparam int A_STRIDE = 13;
    localparam int B_STRIDE = 6;
    localparam int C_STRIDE = 44;
    localparam int CONV_A_BASE = 32'hA00;
    localparam int CONV_B_BASE = 32'hB00;
    localparam int CONV_C_BASE = 32'hC00;
    localparam int CONV_BIAS_BASE = 32'hE00;
    localparam int CONV_IC = 1;
    localparam int CONV_IH = 3;
    localparam int CONV_IW = 5;
    localparam int CONV_OC = 3;
    localparam int CONV_OH = 3;
    localparam int CONV_OW = 5;
    localparam int CONV_KH = 1;
    localparam int CONV_KW = 3;

    logic clk = 0;
    logic rst_n = 0;
    logic [6:0] awaddr = 0, araddr = 0;
    logic awvalid = 0, awready, wvalid = 0, wready;
    logic [31:0] wdata = 0;
    logic [3:0] wstrb = 0;
    logic [1:0] bresp, rresp;
    logic bvalid, bready = 0, arvalid = 0, arready;
    logic [31:0] rdata;
    logic rvalid, rready = 0;
    logic mem_cmd_valid, mem_cmd_ready = 0, mem_cmd_write;
    logic [31:0] mem_cmd_addr, mem_cmd_wdata;
    logic [3:0] mem_cmd_len, mem_cmd_wstrb;
    logic mem_write_valid, mem_write_ready = 0;
    logic [31:0] mem_write_data;
    logic [3:0] mem_write_strb;
    logic mem_write_last;
    logic mem_read_valid = 0, mem_read_ready;
    logic [31:0] mem_read_data = 0;
    logic mem_read_last = 0;
    logic [7:0] ddr [0:MEM_BYTES-1];
    logic read_active = 0;
    logic [31:0] read_addr = 0;
    logic [4:0] read_words = 0;
    logic [31:0] read_cmd_addr_queue [0:4095];
    logic [4:0] read_cmd_words_queue [0:4095];
    integer read_cmd_write_index = 0;
    integer read_cmd_read_index = 0;
    logic [31:0] write_addr = 0;
    logic [1:0] phase = 0;
    integer errors = 0;
    integer conv_a_load_count = 0;
    integer conv_b_load_count = 0;
    integer conv_a_load_sum = 0;
    integer conv_b_load_sum = 0;
    integer conv_vector_delta_count;
    integer conv_vector_delta_sum;
    integer early_prefetch_count = 0;
    integer prefetch_start_count = 0;
    integer two_ahead_count = 0;
    integer gemm_expected [0:M*N-1];
    integer conv_expected [0:CONV_OC*CONV_OH*CONV_OW-1];
    integer rq_expected [0:8];
    integer relu_expected [0:8];
    integer leaky_expected [0:8];
    integer upsample_expected [0:15];
    integer concat_expected [0:8];

    always #5 clk = ~clk;

    always @(posedge clk)
        if (rst_n && dut.engine_busy && !dut.load_a_request &&
            dut.lookahead_ready_tiles != 0)
            early_prefetch_count <= early_prefetch_count + 1;

    always @(posedge clk)
        if (rst_n && !dut.conv_mode &&
            dut.lookahead_descriptor_valid &&
            dut.lookahead_descriptor_ready)
            prefetch_start_count <= prefetch_start_count + 1;

    always @(posedge clk)
        if (rst_n && dut.core_prefetch_valid &&
            dut.lookahead_ready_tiles != 0)
            two_ahead_count <= two_ahead_count + 1;

    always @(posedge clk) begin
        if (rst_n && dut.memio.accept_read &&
            (dut.memio.read_req_len == 0 ||
             dut.memio.read_req_len > 16 ||
             {1'b0, dut.memio.read_req_addr[5:0]} +
                 dut.memio.read_req_len > 64))
            $display("MEMIO invalid request addr=%h len=%0d",
                     dut.memio.read_req_addr, dut.memio.read_req_len);
        if (rst_n && dut.memio.mem_read_valid && dut.memio.mem_read_ready &&
            dut.memio.return_is_direct && !dut.memio.mem_read_last)
            $display("MEMIO direct response missing last");
        if (rst_n && dut.memio.mem_read_valid && dut.memio.mem_read_ready &&
            !dut.memio.return_is_direct && dut.memio.mem_read_last &&
            dut.memio.fill_word_index != 15)
            $display("MEMIO short line response fill_word=%0d",
                     dut.memio.fill_word_index);
        if (rst_n && dut.memio.mem_read_valid && dut.memio.mem_read_ready &&
            !dut.memio.return_is_direct && !dut.memio.mem_read_last &&
            dut.memio.fill_word_index == 15)
            $display("MEMIO long line response");
    end

    function automatic integer aval(input integer row, input integer col);
        aval = ((row * 3 + col * 5 + 4) & 15) - 8;
    endfunction
    function automatic integer bval(input integer row, input integer col);
        bval = ((row * 7 + col * 3 + 2) & 15) - 8;
    endfunction
    function automatic integer conv_aval(input integer y, input integer x);
        conv_aval = ((y * 5 + x * 3 + 6) & 15) - 8;
    endfunction
    function automatic integer conv_bval(input integer channel,
                                          input integer kernel_x);
        conv_bval = ((channel * 5 + kernel_x * 3 + 1) & 15) - 8;
    endfunction
    function automatic logic [31:0] load_word(input logic [31:0] addr);
        load_word = {ddr[addr+3], ddr[addr+2], ddr[addr+1], ddr[addr]};
    endfunction

    task automatic store_word(input logic [31:0] addr,
                              input logic [31:0] data,
                              input logic [3:0] strobe);
        for (integer lane = 0; lane < 4; lane = lane + 1)
            if (strobe[lane]) ddr[addr+lane] = data[lane*8 +: 8];
    endtask

    task automatic store_nibble(input logic [31:0] base,
                                input integer index,
                                input integer value);
        logic [7:0] old_byte;
        begin
            old_byte = ddr[base + (index >> 1)];
            if (index[0])
                ddr[base + (index >> 1)] =
                    {value[3:0], old_byte[3:0]};
            else
                ddr[base + (index >> 1)] =
                    {old_byte[7:4], value[3:0]};
        end
    endtask

    function automatic integer load_nibble(input logic [31:0] base,
                                           input integer index);
        logic signed [3:0] nibble;
        begin
            nibble = index[0] ? ddr[base + (index >> 1)][7:4]
                              : ddr[base + (index >> 1)][3:0];
            load_nibble = $signed(nibble);
        end
    endfunction

    task automatic axi_write(input logic [6:0] addr, input logic [31:0] data);
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            wstrb = 4'hf;
            awvalid = 1;
            wvalid = 1;
            bready = 1;
            @(posedge clk);
            while (!bvalid) @(posedge clk);
            @(negedge clk);
            awvalid = 0;
            wvalid = 0;
            bready = 0;
        end
    endtask

    task automatic axi_read(input logic [6:0] addr, output logic [31:0] data);
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1;
            rready = 1;
            @(posedge clk);
            while (!rvalid) @(posedge clk);
            data = rdata;
            @(negedge clk);
            arvalid = 0;
            rready = 0;
        end
    endtask

    task automatic run_command(input logic [31:0] control,
                               input string label);
        logic [31:0] command_status;
        begin
            axi_write(6'h00, 32'h0);
            axi_write(6'h00, control);
            command_status = 0;
            repeat (20000) begin
                axi_read(6'h04, command_status);
                if (command_status[1] && !command_status[0]) break;
            end
            if (!command_status[1] || command_status[2])
                $fatal(1, "%s failed with status %h", label, command_status);
        end
    endtask

    task automatic run_multim_multik_gemm;
        localparam integer TEST_M = 16;
        localparam integer TEST_N = 8;
        localparam integer TEST_K = 16;
        localparam integer TEST_A_STRIDE = 8;
        localparam integer TEST_B_STRIDE = 4;
        localparam integer TEST_C_STRIDE = 32;
        integer low_value;
        integer high_value;
        integer expected_value;
        integer actual_value;
        begin
            for (integer row = 0; row < TEST_M; row = row + 1)
                for (integer byte_col = 0;
                     byte_col < TEST_A_STRIDE;
                     byte_col = byte_col + 1) begin
                    low_value = aval(row, byte_col*2);
                    high_value = aval(row, byte_col*2+1);
                    ddr[A_BASE + row*TEST_A_STRIDE + byte_col] =
                        ((high_value & 15) << 4) | (low_value & 15);
                end
            for (integer row = 0; row < TEST_K; row = row + 1)
                for (integer byte_col = 0;
                     byte_col < TEST_B_STRIDE;
                     byte_col = byte_col + 1) begin
                    low_value = bval(row, byte_col*2);
                    high_value = bval(row, byte_col*2+1);
                    ddr[B_BASE + row*TEST_B_STRIDE + byte_col] =
                        ((high_value & 15) << 4) | (low_value & 15);
                end
            for (integer index = 0;
                 index < TEST_M*TEST_C_STRIDE;
                 index = index + 1)
                ddr[C_BASE + index] = 0;

            axi_write(6'h08, A_BASE);
            axi_write(6'h0c, B_BASE);
            axi_write(6'h10, C_BASE);
            axi_write(6'h14, TEST_A_STRIDE);
            axi_write(6'h18, TEST_B_STRIDE);
            axi_write(6'h1c, TEST_C_STRIDE);
            axi_write(6'h20, TEST_M);
            axi_write(6'h24, TEST_N);
            axi_write(6'h28, TEST_K);
            run_command(32'h0000_0101, "16x8x16 INT4 GEMM");

            for (integer row = 0; row < TEST_M; row = row + 1)
                for (integer col = 0; col < TEST_N; col = col + 1) begin
                    expected_value = 0;
                    for (integer inner = 0;
                         inner < TEST_K;
                         inner = inner + 1)
                        expected_value = expected_value +
                            aval(row, inner) * bval(inner, col);
                    actual_value = $signed(load_word(
                        C_BASE + row*TEST_C_STRIDE + col*4));
                    if (actual_value != expected_value)
                        $fatal(1,
                               "16x8x16 C[%0d,%0d]=%0d expected %0d",
                               row, col, actual_value, expected_value);
                end
            $display("Multi-M/multi-K INT4 GEMM regression passed");
        end
    endtask

    bcuda_dual_int4_ddr_top dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write, .mem_cmd_addr,
        .mem_cmd_len, .mem_cmd_wdata, .mem_cmd_wstrb,
        .mem_write_valid, .mem_write_ready, .mem_write_data,
        .mem_write_strb, .mem_write_last,
        .mem_read_valid, .mem_read_ready, .mem_read_data, .mem_read_last
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mem_cmd_ready <= 0;
            mem_write_ready <= 0;
            mem_read_valid <= 0;
            mem_read_last <= 0;
            read_active <= 0;
            read_cmd_write_index <= 0;
            read_cmd_read_index <= 0;
            phase <= 0;
        end else begin
            phase <= phase + 1'b1;
            mem_cmd_ready <= phase != 0;
            mem_write_ready <= phase[0];
            mem_read_valid <= read_active;
            mem_read_data <= load_word(read_addr);
            mem_read_last <= read_active && read_words == 1;

            if (read_active && mem_read_ready) begin
                if (read_words == 1) begin
                    if (read_cmd_read_index < read_cmd_write_index) begin
                        read_addr <= read_cmd_addr_queue[read_cmd_read_index];
                        read_words <= read_cmd_words_queue[read_cmd_read_index];
                        read_cmd_read_index <= read_cmd_read_index + 1;
                    end else begin
                        read_active <= 0;
                    end
                end else begin
                    read_addr <= read_addr + 4;
                    read_words <= read_words - 1'b1;
                end
            end else if (!read_active &&
                         read_cmd_read_index < read_cmd_write_index) begin
                read_active <= 1;
                read_addr <= read_cmd_addr_queue[read_cmd_read_index];
                read_words <= read_cmd_words_queue[read_cmd_read_index];
                read_cmd_read_index <= read_cmd_read_index + 1;
            end
            if (mem_cmd_valid && mem_cmd_ready) begin
                if (mem_cmd_write) begin
                    write_addr <= mem_cmd_addr;
                end else begin
                    read_cmd_addr_queue[read_cmd_write_index] <= mem_cmd_addr;
                    read_cmd_words_queue[read_cmd_write_index] <=
                        mem_cmd_len + 1'b1;
                    read_cmd_write_index <= read_cmd_write_index + 1;
                end
            end
            if (mem_write_valid && mem_write_ready) begin
                store_word(write_addr, mem_write_data, mem_write_strb);
                write_addr <= write_addr + 4;
            end
        end
    end

    always_comb begin
        conv_vector_delta_count = 0;
        conv_vector_delta_sum = 0;
        for (integer lane = 0; lane < 8; lane = lane + 1) begin
            if (dut.packed_vector_mask[lane]) begin
                conv_vector_delta_count = conv_vector_delta_count + 1;
                conv_vector_delta_sum = conv_vector_delta_sum +
                    $signed(dut.packed_vector_data[lane*4 +: 4]);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n && dut.packed_vector_valid && dut.packed_vector_ready) begin
            if (dut.packed_vector_matrix) begin
                conv_b_load_count <=
                    conv_b_load_count + conv_vector_delta_count;
                conv_b_load_sum <= conv_b_load_sum + conv_vector_delta_sum;
            end else begin
                conv_a_load_count <=
                    conv_a_load_count + conv_vector_delta_count;
                conv_a_load_sum <= conv_a_load_sum + conv_vector_delta_sum;
            end
        end
    end

    initial begin
        logic [31:0] status;
        logic [31:0] diagnostic;
        integer low_value, high_value, actual;
        gemm_expected = '{
            -4, -28, -100, 36, -52, 196, -68, 36, -116, -12, 92,
            246, 15, -56, -175, 42, 35, 28, 181, -226, 135, -16,
            80, -54, 180, -146, 168, -46, -4, -10, -112, 42, -60,
            -38, 85, 16, -101, -42, -47, 204, -57, 50, -99, 8,
            20, 240, -36, -104, -204, 96, 140, -24, 132, -224, 188
        };
        conv_expected = '{
            10, 9, -27, -47, -22, -15, -35, -7, 69, 35,
            40, 81, 45, 9, -20,
            -3, 16, 25, -30, -25, 22, -33, -40, 1, 2,
            -33, -2, 7, 16, -3,
            -12, -21, -15, 103, -8, -17, 101, 11, -31, -11,
            -6, -33, -27, -21, 34
        };
        rq_expected = '{-8, -8, -1, 0, 3, 7, 7, 7, -8};
        relu_expected = '{0, 0, 0, 0, 1, 3, 6, 7, 0};
        leaky_expected = '{-2, -1, -1, 0, 1, 3, 6, 7, -1};
        upsample_expected = '{
            1, 1, -2, -2,
            1, 1, -2, -2,
            3, 3, 4, 4,
            3, 3, 4, 4
        };
        concat_expected = '{1, 2, 3, -1, -2, -3, 4, 5, 6};
        for (integer index = 0; index < MEM_BYTES; index = index + 1)
            ddr[index] = 8'h00;
        for (integer row = 0; row < M; row = row + 1)
            for (integer byte_col = 0; byte_col < A_STRIDE; byte_col = byte_col + 1) begin
                low_value = aval(row, byte_col*2);
                high_value = byte_col*2+1 < K ? aval(row, byte_col*2+1) : 0;
                ddr[A_BASE + row*A_STRIDE + byte_col] =
                    ((high_value & 15) << 4) | (low_value & 15);
            end
        for (integer row = 0; row < K; row = row + 1)
            for (integer byte_col = 0; byte_col < B_STRIDE; byte_col = byte_col + 1) begin
                low_value = bval(row, byte_col*2);
                high_value = byte_col*2+1 < N ? bval(row, byte_col*2+1) : 0;
                ddr[B_BASE + row*B_STRIDE + byte_col] =
                    ((high_value & 15) << 4) | (low_value & 15);
            end
        for (integer byte_index = 0;
             byte_index < (CONV_IC*CONV_IH*CONV_IW + 1)/2;
             byte_index = byte_index + 1) begin
            low_value = conv_aval((byte_index*2) / CONV_IW,
                                  (byte_index*2) % CONV_IW);
            if (byte_index*2 + 1 < CONV_IC*CONV_IH*CONV_IW)
                high_value = conv_aval((byte_index*2 + 1) / CONV_IW,
                                       (byte_index*2 + 1) % CONV_IW);
            else
                high_value = 0;
            ddr[CONV_A_BASE + byte_index] =
                ((high_value & 15) << 4) | (low_value & 15);
        end
        for (integer byte_index = 0;
             byte_index < (CONV_OC*CONV_IC*CONV_KH*CONV_KW + 1)/2;
             byte_index = byte_index + 1) begin
            low_value = conv_bval((byte_index*2) / CONV_KW,
                                  (byte_index*2) % CONV_KW);
            if (byte_index*2 + 1 < CONV_OC*CONV_IC*CONV_KH*CONV_KW)
                high_value = conv_bval((byte_index*2 + 1) / CONV_KW,
                                       (byte_index*2 + 1) % CONV_KW);
            else
                high_value = 0;
            ddr[CONV_B_BASE + byte_index] =
                ((high_value & 15) << 4) | (low_value & 15);
        end
        store_word(CONV_BIAS_BASE + 0, 32'd3, 4'hf);
        store_word(CONV_BIAS_BASE + 4, -32'sd5, 4'hf);
        store_word(CONV_BIAS_BASE + 8, 32'd7, 4'hf);
        repeat (5) @(posedge clk);
        rst_n = 1;

        // Post-op element counts use the full 32-bit M register. This exact
        // boundary previously truncated to zero in the top-level datapath.
        axi_write(6'h20, 32'h0001_0000);
        axi_write(6'h00, 32'h0000_0103);
        repeat (4) @(posedge clk);
        if (dut.element_count != 32'h0001_0000)
            $fatal(1, "32-bit post-op count truncated to %h",
                   dut.element_count);
        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;

        axi_write(6'h08, A_BASE);
        axi_write(6'h0c, B_BASE);
        axi_write(6'h10, C_BASE);
        axi_write(6'h14, A_STRIDE);
        axi_write(6'h18, B_STRIDE);
        axi_write(6'h1c, C_STRIDE);
        axi_write(6'h20, M);
        axi_write(6'h24, N);
        axi_write(6'h28, K);
        axi_write(6'h00, 32'h0000_0101);

        status = 0;
        repeat (10000) begin
            axi_read(6'h04, status);
            if (status[1] && !status[0]) break;
        end
        if (!status[1] || status[2])
            $fatal(1, "INT4 top status=%h ctl=%0d op=%0d load=%0d writer=%0d count=%0d flush=%0b send=%0b inflight=%0b allocated=%0d ready=%0d outstanding=%0d loader_err=%0b mem_err=%0b readq=%0d returns=%0d ret_direct=%0b fill_word=%0d",
                   status, dut.engine.controller.state, dut.engine.op_state,
                   dut.load_state, dut.writer.state, dut.memio.write_count,
                   dut.memio.write_flush_pending, dut.memio.write_send_active,
                   dut.memio.write_inflight,
                   dut.lookahead_allocated_tiles,
                   dut.lookahead_ready_tiles,
                   dut.lookahead_outstanding_words,
                   dut.lookahead_error, dut.memio.error,
                   dut.memio.read_queue_count, dut.memio.return_count,
                   dut.memio.return_is_direct, dut.memio.fill_word_index);

        for (integer row = 0; row < M; row = row + 1)
            for (integer col = 0; col < N; col = col + 1) begin
                actual = $signed(load_word(C_BASE + row*C_STRIDE + col*4));
                if (actual != gemm_expected[row*N + col]) begin
                    $error("C[%0d,%0d]=%0d expected %0d",
                           row, col, actual, gemm_expected[row*N + col]);
                    errors = errors + 1;
                end
            end
        if (errors != 0) $fatal(1, "INT4 top had %0d mismatches", errors);
        axi_read(7'h7c, diagnostic);
        if (diagnostic != 32'h4449_4132)
            $fatal(1, "diagnostics ABI magic=%h", diagnostic);
        axi_read(7'h40, diagnostic);
        if (diagnostic == 0)
            $fatal(1, "diagnostics total cycle counter is zero");
        axi_read(7'h48, diagnostic);
        if (diagnostic == 0)
            $fatal(1, "diagnostics compute cycle counter is zero");
        axi_read(7'h58, diagnostic);
        if (diagnostic == 0)
            $fatal(1, "diagnostics DDR read command counter is zero");
        axi_read(7'h68, diagnostic);
        if (diagnostic == 0)
            $fatal(1, "diagnostics core row counter is zero");
        axi_read(7'h70, diagnostic);
        if (diagnostic == 0)
            $fatal(1, "diagnostics descriptor counter is zero");
        if (prefetch_start_count < 4)
            $fatal(1, "GEMM queued only %0d look-ahead descriptors",
                   prefetch_start_count);
        if (early_prefetch_count < 1)
            $fatal(1, "look-ahead cache was never ready before compute demand");
        $display("Dual INT4 DDR top golden trace passed: packed 5x11x25 GEMM");
        $display("INT4 prefetch pipeline golden trace passed: started=%0d cache-ready-cycles=%0d two-ahead-cycles=%0d",
                 prefetch_start_count, early_prefetch_count, two_ahead_count);

        run_multim_multik_gemm();

        // Clear the start edge, then launch padded 1x3 Conv2D with bias.
        axi_write(6'h00, 32'h0000_0000);
        axi_write(6'h08, CONV_A_BASE);
        axi_write(6'h0c, CONV_B_BASE);
        axi_write(6'h10, CONV_C_BASE);
        axi_write(6'h14, CONV_IC);
        axi_write(6'h18, CONV_IH);
        axi_write(6'h1c, CONV_IW);
        axi_write(6'h20, CONV_OC);
        axi_write(6'h24, CONV_OH);
        axi_write(6'h28, CONV_OW);
        axi_write(6'h2c, CONV_KH);
        axi_write(6'h30, CONV_KW);
        axi_write(6'h34, 32'h0000_0101);
        axi_write(6'h38, 32'h0000_0100);
        axi_write(6'h3c, CONV_BIAS_BASE);
        axi_write(6'h00, 32'h0000_0105);

        status = 0;
        repeat (10000) begin
            axi_read(6'h04, status);
            if (status[1] && !status[0]) break;
        end
        if (!status[1] || status[2]) begin
            $display("ROW_STREAM active=%0b done=%0b error=%0b row=%0d a_rows=%h b_rows=%h dual=%0b",
                     dut.conv_row_stream.active_r, dut.row_stream_done,
                     dut.row_stream_error, dut.conv_row_stream.consume_row,
                     dut.conv_row_stream.a_rows_r,
                     dut.conv_row_stream.b_rows_r,
                     dut.conv_row_stream.dual_r);
            $display("ROW_STREAM valid a0=%h b0=%h a1=%h b1=%h in=%0b/%0b out=%0b/%0b",
                     dut.conv_row_stream.customer0_a_valid,
                     dut.conv_row_stream.customer0_b_valid,
                     dut.conv_row_stream.customer1_a_valid,
                     dut.conv_row_stream.customer1_b_valid,
                     dut.source_load_valid, dut.row_stream_input_ready,
                     dut.row_stream_output_valid, dut.row_stream_output_ready);
            $display("CONV_LOADER busy=%0b done=%0b error=%0b vector=%0b/%0b matrix=%0b array=%0b row=%0d",
                     dut.conv_loader_busy, dut.conv_loader_done,
                     dut.conv_loader_error, dut.packed_vector_valid,
                     dut.packed_vector_ready, dut.packed_vector_matrix,
                     dut.packed_vector_array, dut.packed_vector_row);
            $display("CORE load_req=%0b group=%0b/%0b prefetch=%0b serviced=%0b",
                     dut.load_a_request, dut.external_group_valid,
                     dut.external_group_ready, dut.core_prefetch_valid,
                     dut.load_request_serviced);
            $fatal(1, "INT4 Conv2D top status=%h ctl=%0d op=%0d load=%0d",
                   status, dut.engine.controller.state, dut.engine.op_state,
                   dut.load_state);
        end
        if (conv_a_load_count != 45 || conv_a_load_sum != -17 ||
            conv_b_load_count != 9 || conv_b_load_sum != -7)
            $fatal(1, "Conv weight-cache trace A count/sum=%0d/%0d B=%0d/%0d",
                   conv_a_load_count, conv_a_load_sum,
                   conv_b_load_count, conv_b_load_sum);
        if (dut.diag_weight_cache_hits != 1 ||
            dut.diag_weight_cache_misses != 0)
            $fatal(1, "Conv weight-cache diagnostics hits=%0d misses=%0d",
                   dut.diag_weight_cache_hits,
                   dut.diag_weight_cache_misses);

        for (integer channel = 0; channel < CONV_OC;
             channel = channel + 1)
            for (integer spatial = 0; spatial < CONV_OH*CONV_OW;
                 spatial = spatial + 1) begin
                actual = $signed(load_word(
                    CONV_C_BASE + (channel*CONV_OH*CONV_OW + spatial)*4));
                if (actual != conv_expected[channel*CONV_OH*CONV_OW + spatial]) begin
                    $error("Conv C[%0d,%0d]=%0d expected %0d",
                           channel, spatial, actual,
                           conv_expected[channel*CONV_OH*CONV_OW + spatial]);
                    errors = errors + 1;
                end
            end
        if (errors != 0)
            $fatal(1, "INT4 Conv2D top had %0d mismatches", errors);
        $display("Dual INT4 DDR top golden trace passed: packed padded Conv2D with bias");

        // INT32 accumulator requantization to an odd packed INT4 length.
        store_word(32'h100, -32'sd20, 4'hf);
        store_word(32'h104, -32'sd8, 4'hf);
        store_word(32'h108, -32'sd1, 4'hf);
        store_word(32'h10c, 32'sd0, 4'hf);
        store_word(32'h110, 32'sd3, 4'hf);
        store_word(32'h114, 32'sd7, 4'hf);
        store_word(32'h118, 32'sd9, 4'hf);
        store_word(32'h11c, 32'sd100, 4'hf);
        store_word(32'h120, -32'sd100, 4'hf);
        axi_write(6'h08, 32'h100);
        axi_write(6'h10, 32'h180);
        axi_write(6'h20, 9);
        axi_write(6'h34, 1);
        axi_write(6'h38, 0);
        axi_write(6'h3c, 0);
        run_command(32'h0000_0103, "INT4 requantize");
        for (integer index = 0; index < 9; index = index + 1)
            if (load_nibble(32'h180, index) != rq_expected[index])
                $fatal(1, "Requant[%0d]=%0d expected %0d", index,
                       load_nibble(32'h180, index), rq_expected[index]);
        if (ddr[32'h184][7:4] != 0)
            $fatal(1, "Requant odd tail high nibble was not zero");
        $display("Dual INT4 DDR top golden trace passed: requantization");

        // Eight lanes are activated together; the ninth element verifies the
        // packed tail and arithmetic right shift in LeakyReLU mode.
        for (integer index = 0; index < 9; index = index + 1)
            store_nibble(32'h200, index,
                index == 0 ? -8 : index == 1 ? -4 : index == 2 ? -1 :
                index == 3 ? 0 : index == 4 ? 1 : index == 5 ? 3 :
                index == 6 ? 6 : index == 7 ? 7 : -3);
        axi_write(6'h08, 32'h200);
        axi_write(6'h10, 32'h240);
        axi_write(6'h20, 9);
        axi_write(6'h38, 2);
        run_command(32'h0000_0109, "INT4 ReLU");
        for (integer index = 0; index < 9; index = index + 1)
            if (load_nibble(32'h240, index) != relu_expected[index])
                $fatal(1, "ReLU[%0d]=%0d expected %0d", index,
                       load_nibble(32'h240, index), relu_expected[index]);
        axi_write(6'h10, 32'h280);
        run_command(32'h0000_0119, "INT4 LeakyReLU");
        for (integer index = 0; index < 9; index = index + 1)
            if (load_nibble(32'h280, index) != leaky_expected[index])
                $fatal(1, "LeakyReLU[%0d]=%0d expected %0d", index,
                       load_nibble(32'h280, index), leaky_expected[index]);
        $display("Dual INT4 DDR top golden trace passed: parallel ReLU/LeakyReLU");

        // MaxPool 2x2, stride 1 over a 3x3 signed INT4 image.
        for (integer index = 0; index < 9; index = index + 1)
            store_nibble(32'h300, index,
                index == 0 ? 1 : index == 1 ? -2 : index == 2 ? 3 :
                index == 3 ? 4 : index == 4 ? 0 : index == 5 ? -5 :
                index == 6 ? 2 : index == 7 ? 7 : -1);
        axi_write(6'h08, 32'h300);
        axi_write(6'h10, 32'h340);
        axi_write(6'h14, 1);
        axi_write(6'h18, 1);
        axi_write(6'h1c, 3);
        axi_write(6'h20, 3);
        axi_write(6'h24, 2);
        axi_write(6'h28, 2);
        axi_write(6'h2c, 2);
        axi_write(6'h30, 2);
        axi_write(6'h34, 32'h0000_0101);
        axi_write(6'h38, 0);
        run_command(32'h0000_0121, "INT4 MaxPool");
        if (load_nibble(32'h340, 0) != 4 ||
            load_nibble(32'h340, 1) != 3 ||
            load_nibble(32'h340, 2) != 7 ||
            load_nibble(32'h340, 3) != 7)
            $fatal(1, "INT4 MaxPool output mismatch: %0d %0d %0d %0d",
                   load_nibble(32'h340, 0), load_nibble(32'h340, 1),
                   load_nibble(32'h340, 2), load_nibble(32'h340, 3));

        // GlobalAveragePool uses the same packed engine and truncates 9/9.
        axi_write(6'h10, 32'h380);
        axi_write(6'h24, 1);
        axi_write(6'h28, 1);
        axi_write(6'h2c, 3);
        axi_write(6'h30, 3);
        run_command(32'h0000_0301, "INT4 GlobalAveragePool");
        if (load_nibble(32'h380, 0) != 1)
            $fatal(1, "INT4 GlobalAveragePool=%0d expected 1",
                   load_nibble(32'h380, 0));
        $display("Dual INT4 DDR top golden trace passed: MaxPool/AveragePool");

        // Nearest-neighbor 2x upsample.
        store_nibble(32'h400, 0, 1);
        store_nibble(32'h400, 1, -2);
        store_nibble(32'h400, 2, 3);
        store_nibble(32'h400, 3, 4);
        axi_write(6'h08, 32'h400);
        axi_write(6'h10, 32'h440);
        axi_write(6'h14, 1);
        axi_write(6'h18, 1);
        axi_write(6'h1c, 2);
        axi_write(6'h20, 2);
        axi_write(6'h24, 4);
        axi_write(6'h28, 4);
        axi_write(6'h34, 32'h0000_0202);
        run_command(32'h0000_0141, "INT4 nearest upsample");
        for (integer index = 0; index < 16; index = index + 1)
            if (load_nibble(32'h440, index) != upsample_expected[index])
                $fatal(1, "Upsample[%0d]=%0d expected %0d", index,
                       load_nibble(32'h440, index),
                       upsample_expected[index]);

        // Three packed NCHW inputs concatenate along the channel axis.
        for (integer index = 0; index < 3; index = index + 1) begin
            store_nibble(32'h500, index, index + 1);
            store_nibble(32'h520, index, -(index + 1));
            store_nibble(32'h540, index, index + 4);
        end
        axi_write(6'h08, 32'h500);
        axi_write(6'h0c, 32'h520);
        axi_write(6'h10, 32'h580);
        axi_write(6'h14, 1);
        axi_write(6'h18, 1);
        axi_write(6'h1c, 1);
        axi_write(6'h20, 3);
        axi_write(6'h24, 1);
        axi_write(6'h28, 3);
        axi_write(6'h34, 1);
        axi_write(6'h38, 1);
        axi_write(6'h3c, 32'h540);
        run_command(32'h0000_0181, "INT4 channel concat");
        for (integer index = 0; index < 9; index = index + 1)
            if (load_nibble(32'h580, index) != concat_expected[index])
                $fatal(1, "Concat[%0d]=%0d expected %0d", index,
                       load_nibble(32'h580, index), concat_expected[index]);
        $display("Dual INT4 DDR top golden trace passed: upsample/concat");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "timeout state=%0d engine=%0b memio=%0b",
               dut.load_state, dut.engine_busy, dut.memio_busy);
    end
endmodule
