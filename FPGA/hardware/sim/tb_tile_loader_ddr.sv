module tb_tile_loader_ddr;
    localparam string DDR_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/tileloader_ddr.hex";
    localparam string A_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/tileloader_A_tile.hex";
    localparam string B_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/tileloader_B_tile.hex";
    localparam string READS_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/tileloader_reads.hex";

    localparam logic [31:0] A_BASE = 32'd101;
    localparam logic [31:0] B_BASE = 32'd299;
    localparam logic [31:0] ROW_STRIDE = 32'd32;

    logic clk = 0;
    logic rst_n = 0;
    logic start;
    logic matrix_select;
    logic [31:0] base_addr;
    logic [31:0] row_stride;
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
    logic [7:0] cache_addr;
    logic [7:0] cache_wdata;

    logic [7:0] ddr [0:1023];
    logic [7:0] a_expected [0:255];
    logic [7:0] b_expected [0:255];
    logic [7:0] a_loaded [0:255];
    logic [7:0] b_loaded [0:255];
    logic [43:0] read_trace [0:511];

    logic pending_read;
    logic [31:0] pending_addr;
    integer expected_read_idx;
    integer i;
    integer errors;

    always #5 clk = ~clk;

    tile_loader_ddr dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .matrix_select(matrix_select),
        .base_addr(base_addr),
        .row_stride(row_stride),
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

    always @(posedge clk) begin
        if (!rst_n) begin
            pending_read <= 1'b0;
            pending_addr <= '0;
            read_data_valid <= 1'b0;
            read_data <= '0;
        end else begin
            read_data_valid <= pending_read;
            read_data <= ddr[pending_addr];

            pending_read <= read_req_valid && read_req_ready;
            if (read_req_valid && read_req_ready) begin
                pending_addr <= read_req_addr;
            end
        end
    end

    always @(posedge clk) begin
        if (read_req_valid && read_req_ready) begin
            if (read_req_addr !== read_trace[expected_read_idx][39:24]) begin
                $error(
                    "Tile loader read address mismatch beat %0d: expected 0x%04h got 0x%08h",
                    expected_read_idx,
                    read_trace[expected_read_idx][39:24],
                    read_req_addr
                );
                errors = errors + 1;
            end
        end

        if (cache_we) begin
            if (cache_matrix !== read_trace[expected_read_idx][40]) begin
                $error("Tile loader matrix mismatch beat %0d", expected_read_idx);
                errors = errors + 1;
            end
            if (cache_addr !== read_trace[expected_read_idx][15:8]) begin
                $error(
                    "Tile loader cache address mismatch beat %0d: expected 0x%02h got 0x%02h",
                    expected_read_idx,
                    read_trace[expected_read_idx][15:8],
                    cache_addr
                );
                errors = errors + 1;
            end
            if (cache_wdata !== read_trace[expected_read_idx][7:0]) begin
                $error(
                    "Tile loader data mismatch beat %0d: expected 0x%02h got 0x%02h",
                    expected_read_idx,
                    read_trace[expected_read_idx][7:0],
                    cache_wdata
                );
                errors = errors + 1;
            end

            if (cache_matrix) begin
                b_loaded[cache_addr] <= cache_wdata;
            end else begin
                a_loaded[cache_addr] <= cache_wdata;
            end
            expected_read_idx = expected_read_idx + 1;
        end
    end

    task automatic run_tile(input logic select_b, input logic [31:0] tile_base);
        begin
            @(negedge clk);
            matrix_select = select_b;
            base_addr = tile_base;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (done == 1'b1);
            if (error) begin
                $error("Tile loader DDR reported error for matrix %0d", select_b);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        expected_read_idx = 0;
        $readmemh(DDR_PATH, ddr);
        $readmemh(A_PATH, a_expected);
        $readmemh(B_PATH, b_expected);
        $readmemh(READS_PATH, read_trace);

        start = 1'b0;
        matrix_select = 1'b0;
        base_addr = '0;
        row_stride = ROW_STRIDE;
        tile_rows = 5'd16;
        tile_cols = 5'd16;
        read_req_ready = 1'b1;

        for (i = 0; i < 256; i = i + 1) begin
            a_loaded[i] = '0;
            b_loaded[i] = '0;
        end

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        run_tile(1'b0, A_BASE);
        run_tile(1'b1, B_BASE);

        repeat (2) @(posedge clk);

        if (expected_read_idx != 512) begin
            $error("Tile loader DDR expected 512 read beats, saw %0d", expected_read_idx);
            errors = errors + 1;
        end

        for (i = 0; i < 256; i = i + 1) begin
            if (a_loaded[i] !== a_expected[i]) begin
                $error("Tile loader DDR A mismatch index %0d: expected 0x%02h got 0x%02h", i, a_expected[i], a_loaded[i]);
                errors = errors + 1;
            end
            if (b_loaded[i] !== b_expected[i]) begin
                $error("Tile loader DDR B mismatch index %0d: expected 0x%02h got 0x%02h", i, b_expected[i], b_loaded[i]);
                errors = errors + 1;
            end
        end

        if (errors != 0) begin
            $fatal(1, "Tile loader DDR golden trace failed with %0d errors", errors);
        end

        $display("Tile loader DDR golden trace passed: 512 read beats");
        $finish;
    end
endmodule
