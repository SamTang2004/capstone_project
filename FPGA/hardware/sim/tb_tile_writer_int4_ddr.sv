module tb_tile_writer_int4_ddr;
    localparam string C_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_tileio_C.hex";
    localparam string BYTES_PATH = "C:/python_projects/capstone_project/FPGA/tests/hardware/golden_traces/int4_tileio_writer_bytes.hex";
    localparam logic [31:0] C_BASE = 32'd151;
    localparam logic [31:0] ROW_STRIDE = 32'd5;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic tile_read_req;
    logic [5:0] tile_read_addr;
    logic tile_read_valid;
    logic signed [3:0] tile_read_data;
    logic write_req_valid;
    logic write_req_last;
    logic write_req_ready;
    logic [31:0] write_req_addr;
    logic [7:0] write_req_data;
    logic [3:0] c_tile [0:63];
    logic [7:0] expected_bytes [0:31];
    integer write_count;
    integer errors;

    always #5 clk = ~clk;

    tile_writer_int4_ddr dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .base_addr(C_BASE),
        .row_stride_bytes(ROW_STRIDE),
        .tile_rows(5'd8),
        .tile_cols(5'd7),
        .busy(busy),
        .done(done),
        .error(error),
        .tile_read_req(tile_read_req),
        .tile_read_addr(tile_read_addr),
        .tile_read_valid(tile_read_valid),
        .tile_read_data(tile_read_data),
        .write_req_valid(write_req_valid),
        .write_req_last(write_req_last),
        .write_req_ready(write_req_ready),
        .write_req_addr(write_req_addr),
        .write_req_data(write_req_data)
    );

    always @(posedge clk) begin
        tile_read_valid <= tile_read_req;
        if (tile_read_req)
            tile_read_data <= c_tile[tile_read_addr];
    end

    always @(posedge clk) begin
        if (write_req_valid && write_req_ready) begin
            integer row;
            integer byte_col;
            logic [31:0] expected_addr;
            row = write_count / 4;
            byte_col = write_count % 4;
            expected_addr = C_BASE + row * ROW_STRIDE + byte_col;
            if (write_req_addr !== expected_addr) begin
                $error("INT4 writer address mismatch write=%0d", write_count);
                errors = errors + 1;
            end
            if (write_req_data !== expected_bytes[write_count]) begin
                $error(
                    "INT4 writer data mismatch write=%0d expected=0x%02h got=0x%02h",
                    write_count, expected_bytes[write_count], write_req_data
                );
                errors = errors + 1;
            end
            if (write_req_last !== (write_count == 31)) begin
                $error("INT4 writer last mismatch write=%0d", write_count);
                errors = errors + 1;
            end
            write_count = write_count + 1;
        end
    end

    initial begin
        $readmemh(C_PATH, c_tile);
        $readmemh(BYTES_PATH, expected_bytes);
        start = 1'b0;
        write_req_ready = 1'b1;
        tile_read_valid = 1'b0;
        tile_read_data = '0;
        write_count = 0;
        errors = 0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        wait (done);
        repeat (2) @(posedge clk);

        if (error) begin
            $error("INT4 tile writer reported error");
            errors = errors + 1;
        end
        if (write_count != 32) begin
            $error("INT4 writer expected 32 packed-byte writes, got %0d", write_count);
            errors = errors + 1;
        end

        if (errors != 0)
            $fatal(1, "INT4 tile writer golden trace failed with %0d errors", errors);

        $display("INT4 tile writer golden trace passed: 56 nibbles -> 32 row-aligned bytes");
        $finish;
    end
endmodule
