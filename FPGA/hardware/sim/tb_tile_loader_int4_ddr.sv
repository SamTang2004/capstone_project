module tb_tile_loader_int4_ddr;
    localparam string DDR_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_tileio_ddr.hex";
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_tileio_A.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_tileio_B.hex";
    localparam logic [31:0] A_BASE = 32'd17;
    localparam logic [31:0] B_BASE = 32'd83;
    localparam logic [31:0] ROW_STRIDE = 32'd5;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start;
    logic matrix_select;
    logic [31:0] base_addr;
    logic [31:0] row_stride_bytes;
    logic [4:0] tile_rows;
    logic [4:0] tile_cols;
    logic busy;
    logic done;
    logic error;
    logic read_req_valid;
    logic read_req_ready;
    logic [31:0] read_req_addr;
    logic read_data_valid;
    logic read_data_ready;
    logic [7:0] read_data;
    logic cache_we;
    logic cache_matrix;
    logic [5:0] cache_addr;
    logic signed [3:0] cache_wdata;

    logic [7:0] ddr [0:255];
    logic [3:0] a_expected [0:63];
    logic [3:0] b_expected [0:63];
    logic [3:0] a_loaded [0:63];
    logic [3:0] b_loaded [0:63];
    logic pending_read;
    logic [31:0] pending_addr;
    integer read_count;
    integer cache_write_count;
    integer errors;
    integer i;

    always #5 clk = ~clk;

    tile_loader_int4_ddr dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .matrix_select(matrix_select),
        .base_addr(base_addr),
        .row_stride_bytes(row_stride_bytes),
        .tile_rows(tile_rows),
        .tile_cols(tile_cols),
        .busy(busy),
        .done(done),
        .error(error),
        .read_req_valid(read_req_valid),
        .read_req_ready(read_req_ready),
        .read_req_addr(read_req_addr),
        .read_data_valid(read_data_valid),
        .read_data_ready(read_data_ready),
        .read_data(read_data),
        .cache_we(cache_we),
        .cache_matrix(cache_matrix),
        .cache_addr(cache_addr),
        .cache_wdata(cache_wdata)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pending_read <= 1'b0;
            pending_addr <= '0;
            read_data_valid <= 1'b0;
            read_data <= '0;
        end else begin
            read_data_valid <= pending_read;
            read_data <= ddr[pending_addr];
            pending_read <= read_req_valid && read_req_ready;
            if (read_req_valid && read_req_ready)
                pending_addr <= read_req_addr;
        end
    end

    always @(posedge clk) begin
        if (read_req_valid && read_req_ready) begin
            integer local_index;
            integer expected_row;
            integer expected_byte_col;
            logic [31:0] expected_addr;
            local_index = read_count % 32;
            expected_row = local_index / 4;
            expected_byte_col = local_index % 4;
            expected_addr = (read_count < 32 ? A_BASE : B_BASE) +
                            expected_row * ROW_STRIDE + expected_byte_col;
            if (read_req_addr !== expected_addr) begin
                $error(
                    "INT4 loader address mismatch read=%0d expected=0x%08h got=0x%08h",
                    read_count, expected_addr, read_req_addr
                );
                errors = errors + 1;
            end
            read_count = read_count + 1;
        end

        if (cache_we) begin
            if (cache_matrix)
                b_loaded[cache_addr] = cache_wdata;
            else
                a_loaded[cache_addr] = cache_wdata;
            cache_write_count = cache_write_count + 1;
        end
    end

    task automatic run_tile(input logic select_b, input logic [31:0] selected_base);
        begin
            @(negedge clk);
            matrix_select = select_b;
            base_addr = selected_base;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (done);
            if (error) begin
                $error("INT4 tile loader reported error for matrix %0d", select_b);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        $readmemh(DDR_PATH, ddr);
        $readmemh(A_PATH, a_expected);
        $readmemh(B_PATH, b_expected);

        start = 1'b0;
        matrix_select = 1'b0;
        base_addr = '0;
        row_stride_bytes = ROW_STRIDE;
        tile_rows = 5'd8;
        tile_cols = 5'd8;
        read_req_ready = 1'b1;
        read_count = 0;
        cache_write_count = 0;
        errors = 0;
        for (i = 0; i < 64; i = i + 1) begin
            a_loaded[i] = '0;
            b_loaded[i] = '0;
        end

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        run_tile(1'b0, A_BASE);
        run_tile(1'b1, B_BASE);
        repeat (2) @(posedge clk);

        if (read_count != 64) begin
            $error("INT4 loader expected 64 packed-byte reads, got %0d", read_count);
            errors = errors + 1;
        end
        if (cache_write_count != 128) begin
            $error("INT4 loader expected 128 nibble writes, got %0d", cache_write_count);
            errors = errors + 1;
        end

        for (i = 0; i < 64; i = i + 1) begin
            if (a_loaded[i] !== a_expected[i]) begin
                $error("INT4 loader A mismatch index=%0d", i);
                errors = errors + 1;
            end
            if (b_loaded[i] !== b_expected[i]) begin
                $error("INT4 loader B mismatch index=%0d", i);
                errors = errors + 1;
            end
        end

        if (errors != 0)
            $fatal(1, "INT4 tile loader golden trace failed with %0d errors", errors);

        $display("INT4 tile loader golden trace passed: 64 DDR bytes -> 128 nibbles");
        $finish;
    end
endmodule
