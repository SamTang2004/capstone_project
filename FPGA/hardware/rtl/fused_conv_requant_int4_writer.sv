`timescale 1ns/1ps
// Fuses Conv2D result writeback, requantization, and optional activation.
// One 8x16 result tile is retained so channel-major NCHW INT4 values can be
// packed into aligned 32-bit writes without an INT32 DDR intermediate.
module fused_conv_requant_int4_writer (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    output logic               input_ready,
    input  logic               input_array,
    input  logic [5:0]         input_addr,
    input  logic signed [31:0] input_data,
    input  logic               input_last,
    input  logic [31:0]        output_base_addr,
    input  logic [15:0]        m0,
    input  logic [15:0]        n0,
    input  logic [3:0]         m_len,
    input  logic [4:0]         n_cols,
    input  logic [15:0]        out_h,
    input  logic [15:0]        out_w,
    input  logic signed [31:0] multiplier,
    input  logic [5:0]         right_shift,
    input  logic signed [31:0] output_zero_point,
    input  logic               activation_enable,
    input  logic               leaky_mode,
    input  logic [2:0]         leaky_shift,
    output logic               write_req_valid,
    input  logic               write_req_ready,
    output logic [31:0]        write_req_addr,
    output logic [31:0]        write_req_data,
    output logic [3:0]         write_req_strb,
    output logic               write_req_last
);
    logic signed [3:0] quantized_value;
    logic quantized_valid;
    logic tag_valid [0:3];
    logic tag_array [0:3];
    logic [5:0] tag_addr [0:3];
    logic tag_last [0:3];
    logic signed [3:0] tile_data [0:127];
    logic input_closed_r, draining_r;
    logic [1:0] prepare_drain_r;
    logic [4:0] drain_channel_r;
    logic [31:0] spatial_r;
    logic [31:0] channel_product_r;
    logic [31:0] channel_stride_r;
    logic [31:0] write_addr_r;
    logic [15:0] tile_m0_r, tile_n0_r;
    logic [3:0] tile_m_len_r;
    logic [4:0] tile_n_cols_r;
    logic [15:0] tile_out_h_r, tile_out_w_r;
    logic signed [3:0] activated_value;

    assign input_ready = !input_closed_r && !draining_r;
    assign activated_value =
        activation_enable && quantized_value < 0 ?
            (leaky_mode ? quantized_value >>> leaky_shift : 4'sd0) :
            quantized_value;

    requantize_int4_pipeline requantize (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid && input_ready),
        .input_value(input_data),
        .multiplier(multiplier),
        .right_shift(right_shift),
        .output_zero_point(output_zero_point),
        .output_valid(quantized_valid),
        .output_value(quantized_value)
    );

    assign write_req_valid = draining_r;
    assign write_req_addr = write_addr_r;
    assign write_req_last = draining_r &&
                            drain_channel_r + 1'b1 >= tile_n_cols_r;

    always_comb begin
        write_req_data = '0;
        write_req_strb = '0;
        for (int lane = 0; lane < 8; lane++) begin
            if (lane < tile_m_len_r) begin
                write_req_data[lane*4 +: 4] =
                    tile_data[{drain_channel_r[3],
                               lane[2:0],
                               drain_channel_r[2:0]}];
                write_req_strb[lane >> 1] = 1'b1;
            end
        end
    end

    integer tag_stage;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            input_closed_r <= 1'b0;
            draining_r <= 1'b0;
            prepare_drain_r <= '0;
            drain_channel_r <= '0;
            spatial_r <= '0;
            channel_product_r <= '0;
            channel_stride_r <= '0;
            write_addr_r <= '0;
            tile_m0_r <= '0;
            tile_n0_r <= '0;
            tile_m_len_r <= '0;
            tile_n_cols_r <= '0;
            tile_out_h_r <= '0;
            tile_out_w_r <= '0;
            for (tag_stage = 0; tag_stage < 4;
                 tag_stage = tag_stage + 1) begin
                tag_valid[tag_stage] <= 1'b0;
                tag_array[tag_stage] <= 1'b0;
                tag_addr[tag_stage] <= '0;
                tag_last[tag_stage] <= 1'b0;
            end
        end else begin
            tag_valid[0] <= input_valid && input_ready;
            if (input_valid && input_ready) begin
                tag_array[0] <= input_array;
                tag_addr[0] <= input_addr;
                tag_last[0] <= input_last;
                if (input_last) begin
                    input_closed_r <= 1'b1;
                    tile_m0_r <= m0;
                    tile_n0_r <= n0;
                    tile_m_len_r <= m_len;
                    tile_n_cols_r <= n_cols;
                    tile_out_h_r <= out_h;
                    tile_out_w_r <= out_w;
                end
            end
            for (tag_stage = 1; tag_stage < 4;
                 tag_stage = tag_stage + 1) begin
                tag_valid[tag_stage] <= tag_valid[tag_stage-1];
                tag_array[tag_stage] <= tag_array[tag_stage-1];
                tag_addr[tag_stage] <= tag_addr[tag_stage-1];
                tag_last[tag_stage] <= tag_last[tag_stage-1];
            end

            if (quantized_valid && tag_valid[3]) begin
                tile_data[{tag_array[3], tag_addr[3]}] <= activated_value;
                if (tag_last[3]) begin
                    spatial_r <= tile_out_h_r * tile_out_w_r;
                    drain_channel_r <= '0;
                    prepare_drain_r <= 2'd1;
                end
            end

            if (prepare_drain_r == 2'd1) begin
                channel_product_r <= tile_n0_r * spatial_r;
                prepare_drain_r <= 2'd2;
            end else if (prepare_drain_r == 2'd2) begin
                write_addr_r <= output_base_addr +
                                ((channel_product_r + tile_m0_r) >> 1);
                channel_stride_r <= spatial_r >> 1;
                prepare_drain_r <= '0;
                draining_r <= 1'b1;
            end

            if (write_req_valid && write_req_ready) begin
                if (write_req_last) begin
                    draining_r <= 1'b0;
                    input_closed_r <= 1'b0;
                    drain_channel_r <= '0;
                end else begin
                    drain_channel_r <= drain_channel_r + 1'b1;
                    write_addr_r <= write_addr_r + channel_stride_r;
                end
            end
        end
    end
endmodule
