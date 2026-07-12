module tb_tile_writer_ddr;
    localparam string C_TILE_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/tilewriter_C_tile.hex";

    localparam logic [31:0] C_BASE = 32'd256;
    localparam logic [31:0] ROW_STRIDE = 32'd64;

    logic clk = 0;
    logic rst_n = 0;
    logic start;
    logic [31:0] base_addr;
    logic [31:0] row_stride;
    logic [4:0] tile_rows;
    logic [4:0] tile_cols;
    logic busy;
    logic done;
    logic error;
    logic tile_read_req;
    logic [7:0] tile_read_addr;
    logic tile_read_valid;
    logic signed [31:0] tile_read_data;
    logic write_req_valid;
    logic write_req_ready;
    logic [31:0] write_req_addr;
    logic signed [31:0] write_req_data;

    logic [31:0] c_tile [0:255];

    integer expected_write_idx;
    integer errors;

    always #5 clk = ~clk;

    always @(posedge clk) begin
        tile_read_valid <= tile_read_req;
        if (tile_read_req) tile_read_data <= c_tile[tile_read_addr];
    end

    tile_writer_ddr dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .base_addr(base_addr),
        .row_stride(row_stride),
        .tile_rows(tile_rows),
        .tile_cols(tile_cols),
        .busy(busy),
        .done(done),
        .error(error),
        .tile_read_req(tile_read_req),
        .tile_read_addr(tile_read_addr),
        .tile_read_valid(tile_read_valid),
        .tile_read_data(tile_read_data),
        .write_req_valid(write_req_valid),
        .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr),
        .write_req_data(write_req_data)
    );

    always @(posedge clk) begin
        if (write_req_valid && write_req_ready) begin
            logic [31:0] expected_addr;
            logic [7:0] expected_tile_addr;
            expected_tile_addr = expected_write_idx[7:0];
            expected_addr = C_BASE + ((expected_write_idx / 16) * ROW_STRIDE) + ((expected_write_idx % 16) * 4);

            if (write_req_addr !== expected_addr) begin
                $error("Tile writer address mismatch beat %0d: expected 0x%08h got 0x%08h",
                    expected_write_idx, expected_addr, write_req_addr);
                errors = errors + 1;
            end
            if (tile_read_addr !== expected_tile_addr) begin
                $error("Tile writer tile address mismatch beat %0d: expected 0x%02h got 0x%02h",
                    expected_write_idx, expected_tile_addr, tile_read_addr);
                errors = errors + 1;
            end
            if (write_req_data !== c_tile[expected_tile_addr]) begin
                $error("Tile writer data mismatch beat %0d: expected 0x%08h got 0x%08h",
                    expected_write_idx, c_tile[expected_tile_addr], write_req_data);
                errors = errors + 1;
            end

            expected_write_idx = expected_write_idx + 1;
        end
    end

    initial begin
        errors = 0;
        expected_write_idx = 0;
        $readmemh(C_TILE_PATH, c_tile);

        start = 1'b0;
        base_addr = C_BASE;
        row_stride = ROW_STRIDE;
        tile_rows = 5'd16;
        tile_cols = 5'd16;
        write_req_ready = 1'b1;
        tile_read_valid = 1'b0;
        tile_read_data = '0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done == 1'b1);
        if (error) begin
            $error("Tile writer DDR reported error");
            errors = errors + 1;
        end

        repeat (2) @(posedge clk);

        if (expected_write_idx != 256) begin
            $error("Tile writer DDR expected 256 write beats, saw %0d", expected_write_idx);
            errors = errors + 1;
        end

        if (errors != 0) begin
            $fatal(1, "Tile writer DDR golden trace failed with %0d errors", errors);
        end

        $display("Tile writer DDR golden trace passed: 256 byte-addressed write beats");
        $finish;
    end
endmodule
