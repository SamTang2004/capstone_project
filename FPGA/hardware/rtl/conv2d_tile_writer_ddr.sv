// Restores NCHW layout and overlaps accumulator BRAM latency with AXI writes.
module conv2d_tile_writer_ddr #(
    parameter int TILE_SIZE = 16,
    parameter int FIFO_DEPTH = 8
) (
    input logic clk, input logic rst_n, input logic start, input logic conv_mode,
    input logic [31:0] base_addr, input logic [31:0] row_stride,
    input logic [31:0] output_base_addr, input logic [15:0] out_h, input logic [15:0] out_w,
    input logic [15:0] m0, input logic [15:0] n0, input logic [4:0] tile_rows, input logic [4:0] tile_cols,
    output logic busy, output logic done, output logic error,
    output logic tile_read_req, output logic [7:0] tile_read_addr,
    input logic tile_read_valid, input logic signed [31:0] tile_read_data,
    output logic write_req_valid, output logic write_req_last, input logic write_req_ready,
    output logic [31:0] write_req_addr, output logic signed [31:0] write_req_data
);
    localparam int PTR_W = $clog2(FIFO_DEPTH);
    typedef enum logic [2:0] {
        IDLE, SETUP_PLANE, SETUP_CHANNEL, SETUP_BASE, STREAM, FINISH, FAIL
    } state_t;
    state_t state;

    logic conv_r;
    logic [4:0] issue_row, issue_col, rows_r, cols_r;
    logic [31:0] base_r, stride_r, issue_row_base, issue_addr;
    logic [31:0] plane_elements_r, plane_bytes_r;
    logic [47:0] channel_product_r;
    logic [15:0] m0_r, n0_r;
    logic issue_done;
    logic setup_mul_start, setup_mul_busy, setup_mul_done;
    logic [31:0] setup_mul_lhs;
    logic [15:0] setup_mul_rhs;
    logic [47:0] setup_mul_product;

    logic [31:0] metadata_addr [0:FIFO_DEPTH-1];
    logic metadata_last [0:FIFO_DEPTH-1];
    logic [PTR_W-1:0] metadata_write_ptr, metadata_read_ptr;
    logic [PTR_W:0] outstanding_count;

    logic signed [31:0] data_fifo_data [0:FIFO_DEPTH-1];
    logic [31:0] data_fifo_addr [0:FIFO_DEPTH-1];
    logic data_fifo_last [0:FIFO_DEPTH-1];
    logic [PTR_W-1:0] data_write_ptr, data_read_ptr;
    logic [PTR_W:0] data_count;

    wire issue_is_last = issue_row == rows_r - 1'b1 && issue_col == cols_r - 1'b1;
    wire issue_run_last = conv_r ? issue_row == rows_r - 1'b1 : issue_is_last;
    wire [PTR_W+1:0] reserved_count = data_count + outstanding_count;
    wire issue_fire = state == STREAM && !issue_done && reserved_count < FIFO_DEPTH;
    wire write_fire = write_req_valid && write_req_ready;

    assign busy = state != IDLE && state != FINISH && state != FAIL;
    assign tile_read_req = issue_fire;
    assign tile_read_addr = issue_row * TILE_SIZE + issue_col;
    assign write_req_valid = state == STREAM && data_count != 0;
    assign write_req_addr = data_fifo_addr[data_read_ptr];
    assign write_req_data = data_fifo_data[data_read_ptr];
    assign write_req_last = data_fifo_last[data_read_ptr];

    lut_mul32x16_pipeline setup_multiplier (
        .clk(clk),
        .rst_n(rst_n),
        .start(setup_mul_start),
        .signed_mode(1'b0),
        .lhs(setup_mul_lhs),
        .rhs(setup_mul_rhs),
        .busy(setup_mul_busy),
        .done(setup_mul_done),
        .product(setup_mul_product)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            conv_r <= 1'b0;
            issue_row <= '0;
            issue_col <= '0;
            rows_r <= '0;
            cols_r <= '0;
            base_r <= '0;
            stride_r <= '0;
            issue_row_base <= '0;
            issue_addr <= '0;
            plane_elements_r <= '0;
            plane_bytes_r <= '0;
            channel_product_r <= '0;
            m0_r <= '0;
            n0_r <= '0;
            issue_done <= 1'b0;
            setup_mul_start <= 1'b0;
            setup_mul_lhs <= '0;
            setup_mul_rhs <= '0;
            metadata_write_ptr <= '0;
            metadata_read_ptr <= '0;
            outstanding_count <= '0;
            data_write_ptr <= '0;
            data_read_ptr <= '0;
            data_count <= '0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            done <= 1'b0;
            setup_mul_start <= 1'b0;
            unique case (state)
                IDLE: begin
                    error <= 1'b0;
                    issue_row <= '0;
                    issue_col <= '0;
                    issue_done <= 1'b0;
                    metadata_write_ptr <= '0;
                    metadata_read_ptr <= '0;
                    outstanding_count <= '0;
                    data_write_ptr <= '0;
                    data_read_ptr <= '0;
                    data_count <= '0;
                    if (start) begin
                        conv_r <= conv_mode;
                        rows_r <= tile_rows;
                        cols_r <= tile_cols;
                        base_r <= conv_mode ? output_base_addr : base_addr;
                        stride_r <= row_stride;
                        m0_r <= m0;
                        n0_r <= n0;
                        if (tile_rows == 0 || tile_rows > TILE_SIZE ||
                            tile_cols == 0 || tile_cols > TILE_SIZE ||
                            (conv_mode && (out_h == 0 || out_w == 0))) begin
                            error <= 1'b1;
                            state <= FAIL;
                        end else if (conv_mode) begin
                            setup_mul_lhs <= {16'd0, out_h};
                            setup_mul_rhs <= out_w;
                            setup_mul_start <= 1'b1;
                            state <= SETUP_PLANE;
                        end else begin
                            issue_row_base <= base_addr;
                            issue_addr <= base_addr;
                            state <= STREAM;
                        end
                    end
                end

                SETUP_PLANE: if (setup_mul_done) begin
                    plane_elements_r <= setup_mul_product[31:0];
                    plane_bytes_r <= setup_mul_product[31:0] << 2;
                    setup_mul_lhs <= setup_mul_product[31:0];
                    setup_mul_rhs <= n0_r;
                    setup_mul_start <= 1'b1;
                    state <= SETUP_CHANNEL;
                end
                SETUP_CHANNEL: if (setup_mul_done) begin
                    channel_product_r <= setup_mul_product;
                    state <= SETUP_BASE;
                end
                SETUP_BASE: begin
                    issue_row_base <= base_r + ((channel_product_r[31:0] + m0_r) << 2);
                    issue_addr <= base_r + ((channel_product_r[31:0] + m0_r) << 2);
                    state <= STREAM;
                end

                STREAM: begin
                    if (issue_fire) begin
                        metadata_addr[metadata_write_ptr] <= issue_addr;
                        metadata_last[metadata_write_ptr] <= issue_run_last;
                        metadata_write_ptr <= metadata_write_ptr + 1'b1;
                        if (issue_is_last) begin
                            issue_done <= 1'b1;
                        end else if (conv_r && issue_row == rows_r - 1'b1) begin
                            issue_row <= '0;
                            issue_col <= issue_col + 1'b1;
                            issue_row_base <= issue_row_base + plane_bytes_r;
                            issue_addr <= issue_row_base + plane_bytes_r;
                        end else if (!conv_r && issue_col == cols_r - 1'b1) begin
                            issue_row <= issue_row + 1'b1;
                            issue_col <= '0;
                            issue_row_base <= issue_row_base + stride_r;
                            issue_addr <= issue_row_base + stride_r;
                        end else begin
                            if (conv_r) issue_row <= issue_row + 1'b1;
                            else issue_col <= issue_col + 1'b1;
                            issue_addr <= issue_addr + 4;
                        end
                    end

                    if (tile_read_valid) begin
                        data_fifo_data[data_write_ptr] <= tile_read_data;
                        data_fifo_addr[data_write_ptr] <= metadata_addr[metadata_read_ptr];
                        data_fifo_last[data_write_ptr] <= metadata_last[metadata_read_ptr];
                        data_write_ptr <= data_write_ptr + 1'b1;
                        metadata_read_ptr <= metadata_read_ptr + 1'b1;
                    end
                    if (write_fire) data_read_ptr <= data_read_ptr + 1'b1;

                    unique case ({issue_fire, tile_read_valid})
                        2'b10: outstanding_count <= outstanding_count + 1'b1;
                        2'b01: outstanding_count <= outstanding_count - 1'b1;
                        default: begin end
                    endcase
                    unique case ({tile_read_valid, write_fire})
                        2'b10: data_count <= data_count + 1'b1;
                        2'b01: data_count <= data_count - 1'b1;
                        default: begin end
                    endcase

                    if (write_fire && data_fifo_last[data_read_ptr] &&
                        issue_done && outstanding_count == 0 && data_count == 1)
                        state <= FINISH;
                end

                FINISH: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                FAIL: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
