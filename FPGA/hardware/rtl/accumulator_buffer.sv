module accumulator_buffer #(
    parameter int TILE_SIZE = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    op_valid,
    output logic                    op_ready,
    input  logic                    op_clear,
    input  logic [7:0]              op_addr,
    input  logic signed [31:0]      op_data,
    output logic signed [31:0]      op_result,
    output logic                    op_commit,

    input  logic                    seed_valid,
    output logic                    seed_ready,
    input  logic [7:0]              seed_addr,
    input  logic signed [31:0]      seed_data,

    input  logic                    read_req,
    input  logic [7:0]              read_addr,
    output logic                    read_valid,
    output logic signed [31:0]      read_data
);
    localparam int ENTRY_COUNT = TILE_SIZE * TILE_SIZE;

    // During accumulation, port B reads the next address while port A commits
    // the previous result. During writeback, port B serves TileWriter instead.
    logic op_stage_valid;
    logic op_stage_clear;
    logic [7:0] op_stage_addr;
    logic signed [31:0] op_stage_data;
    logic signed [31:0] port_b_data;
    logic signed [31:0] commit_result;
    logic op_accept;
    logic mem_write_enable;
    logic [7:0] mem_write_addr;
    logic signed [31:0] mem_write_data;
    logic [7:0] mem_read_addr;
    logic [3:0] read_pending_pipe;
    logic signed [31:0] read_data_stage;
    logic signed [31:0] read_data_stage2;
    logic signed [31:0] read_data_stage3;
    logic signed [31:0] read_data_r;

    assign op_ready = !seed_valid && !(op_stage_valid && op_valid && op_stage_addr == op_addr);
    assign op_accept = op_valid && op_ready;
    assign seed_ready = !op_stage_valid && !op_valid;
    assign commit_result = op_stage_clear ? op_stage_data : port_b_data + op_stage_data;
    assign read_data = read_data_r;
    assign mem_write_enable = op_stage_valid || (seed_valid && seed_ready);
    assign mem_write_addr = op_stage_valid ? op_stage_addr : seed_addr;
    assign mem_write_data = op_stage_valid ? commit_result : seed_data;
    assign mem_read_addr = op_accept ? op_addr : read_addr;

    xpm_memory_sdpram #(
        .MEMORY_SIZE(ENTRY_COUNT * 32),
        .MEMORY_PRIMITIVE("block"),
        .CLOCKING_MODE("common_clock"),
        .MEMORY_INIT_FILE("none"),
        .USE_MEM_INIT(0),
        .WRITE_DATA_WIDTH_A(32),
        .BYTE_WRITE_WIDTH_A(32),
        .ADDR_WIDTH_A(8),
        .READ_DATA_WIDTH_B(32),
        .ADDR_WIDTH_B(8),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) accumulator_memory (
        .sleep(1'b0),
        .clka(clk),
        .ena(mem_write_enable),
        .wea(mem_write_enable),
        .addra(mem_write_addr),
        .dina(mem_write_data),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(clk),
        .rstb(!rst_n),
        .enb(1'b1),
        .regceb(1'b1),
        .addrb(mem_read_addr),
        .doutb(port_b_data),
        .sbiterrb(),
        .dbiterrb()
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            op_stage_valid <= 1'b0;
            op_stage_clear <= 1'b0;
            op_stage_addr <= '0;
            op_stage_data <= '0;
            op_result <= '0;
            op_commit <= 1'b0;
            read_pending_pipe <= '0;
            read_data_stage <= '0;
            read_data_stage2 <= '0;
            read_data_stage3 <= '0;
            read_data_r <= '0;
            read_valid <= 1'b0;
        end else begin
            op_commit <= op_stage_valid;
            if (op_stage_valid) begin
                op_result <= commit_result;
            end

            op_stage_valid <= op_accept;
            if (op_accept) begin
                op_stage_clear <= op_clear;
                op_stage_addr <= op_addr;
                op_stage_data <= op_data;
            end

            read_data_stage <= port_b_data;
            read_data_stage2 <= read_data_stage;
            read_data_stage3 <= read_data_stage2;
            read_data_r <= read_data_stage3;
            read_pending_pipe <= {read_pending_pipe[2:0], read_req && !op_accept};
            read_valid <= read_pending_pipe[3];
        end
    end
endmodule
