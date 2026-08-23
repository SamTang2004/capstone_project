module ping_pong_tile_buffer #(
    parameter int DATA_WIDTH = 64,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = $clog2(DEPTH),
    parameter int LENGTH_WIDTH = $clog2(DEPTH + 1)
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    write_start,
    output logic                    write_start_ready,
    input  logic [LENGTH_WIDTH-1:0] write_length,
    input  logic                    write_valid,
    output logic                    write_ready,
    input  logic [DATA_WIDTH-1:0]   write_data,
    output logic                    write_busy,
    output logic                    write_done,

    input  logic                    read_start,
    output logic                    read_start_ready,
    output logic                    read_available,
    output logic                    read_valid,
    input  logic                    read_ready,
    output logic [DATA_WIDTH-1:0]   read_data,
    output logic                    read_last,
    output logic                    read_busy,
    output logic                    read_done,

    output logic                    error
);
    logic [1:0] bank_ready;
    logic [LENGTH_WIDTH-1:0] bank_length [0:1];
    logic write_bank, read_bank;
    logic [ADDR_WIDTH-1:0] write_index, read_index;
    logic [LENGTH_WIDTH-1:0] active_write_length, active_read_length;
    logic read_issue_done;
    logic bank0_write_enable, bank1_write_enable;
    logic bank0_read_enable, bank1_read_enable;
    logic [DATA_WIDTH-1:0] bank0_read_data, bank1_read_data;
    logic read_issue;

    assign write_start_ready = !write_busy && !bank_ready[write_bank] &&
        !(read_busy && read_bank == write_bank);
    assign write_ready = write_busy;
    assign read_available = bank_ready[read_bank] && !read_busy;
    assign read_start_ready = read_available;
    assign read_issue = read_busy && (!read_valid || read_ready) && !read_issue_done;
    assign bank0_write_enable = write_valid && write_ready && !write_bank;
    assign bank1_write_enable = write_valid && write_ready && write_bank;
    assign bank0_read_enable = read_issue && !read_bank;
    assign bank1_read_enable = read_issue && read_bank;
    assign read_data = read_bank ? bank1_read_data : bank0_read_data;

    simple_dual_port_bram #(
        .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)
    ) bank0_ram (
        .clk(clk),
        .write_enable(bank0_write_enable), .write_addr(write_index), .write_data(write_data),
        .read_enable(bank0_read_enable), .read_addr(read_index), .read_data(bank0_read_data)
    );

    simple_dual_port_bram #(
        .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)
    ) bank1_ram (
        .clk(clk),
        .write_enable(bank1_write_enable), .write_addr(write_index), .write_data(write_data),
        .read_enable(bank1_read_enable), .read_addr(read_index), .read_data(bank1_read_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            bank_ready <= '0;
            bank_length[0] <= '0;
            bank_length[1] <= '0;
            write_bank <= 1'b0;
            read_bank <= 1'b0;
            write_index <= '0;
            read_index <= '0;
            active_write_length <= '0;
            active_read_length <= '0;
            write_busy <= 1'b0;
            read_busy <= 1'b0;
            write_done <= 1'b0;
            read_done <= 1'b0;
            read_valid <= 1'b0;
            read_last <= 1'b0;
            read_issue_done <= 1'b0;
            error <= 1'b0;
        end else begin
            write_done <= 1'b0;
            read_done <= 1'b0;

            if (write_start && write_start_ready) begin
                if (write_length == 0 || write_length > DEPTH) begin
                    error <= 1'b1;
                end else begin
                    write_busy <= 1'b1;
                    write_index <= '0;
                    active_write_length <= write_length;
                end
            end

            if (write_valid && write_ready) begin
                if (write_index == active_write_length - 1'b1) begin
                    bank_ready[write_bank] <= 1'b1;
                    bank_length[write_bank] <= active_write_length;
                    write_busy <= 1'b0;
                    write_done <= 1'b1;
                    write_bank <= ~write_bank;
                end else begin
                    write_index <= write_index + 1'b1;
                end
            end

            if (read_start && read_start_ready) begin
                read_busy <= 1'b1;
                read_index <= '0;
                active_read_length <= bank_length[read_bank];
                read_valid <= 1'b0;
                read_last <= 1'b0;
                read_issue_done <= 1'b0;
            end

            if (read_busy && (!read_valid || read_ready)) begin
                if (!read_issue_done) begin
                    read_valid <= 1'b1;
                    read_last <= read_index == active_read_length - 1'b1;
                    if (read_index == active_read_length - 1'b1)
                        read_issue_done <= 1'b1;
                    else
                        read_index <= read_index + 1'b1;
                end else if (read_valid && read_ready) begin
                    read_valid <= 1'b0;
                    read_last <= 1'b0;
                    read_busy <= 1'b0;
                    read_done <= 1'b1;
                    bank_ready[read_bank] <= 1'b0;
                    read_bank <= ~read_bank;
                end
            end
        end
    end
endmodule
