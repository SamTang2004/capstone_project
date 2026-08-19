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

    // Port A performs the serialized accumulation/bias read-modify-write.
    // Port B is the registered TileWriter read port. This synchronous
    // two-port template maps one accumulator tile into a RAMB18.
    (* ram_style = "block", ram_decomp = "power" *) logic signed [31:0] mem [0:ENTRY_COUNT-1];
    logic                    op_pending;
    logic                    pending_clear;
    logic [7:0]              pending_addr;
    logic signed [31:0]      pending_data;
    logic signed [31:0]      pending_old_value;
    logic signed [31:0]      pending_result;
    logic                    port_a_write;
    logic [7:0]              port_a_addr;
    logic signed [31:0]      port_a_data;
    logic [3:0]              read_pending_pipe;
    logic [7:0] read_addr_r;
    logic signed [31:0] read_mem_data;
    // Keep this second response stage outside the RAM primitive. Vivado may
    // absorb read_mem_data into DOA_REG, but this always-clocked register must
    // remain separate so TileWriter state cannot gate the physical response.
    (* dont_touch = "yes" *) logic signed [31:0] read_data_r;

    assign op_ready = !op_pending;
    // Used by Conv2D bias preload while the regular accumulator port is idle.
    assign seed_ready = !op_pending;
    assign pending_result = pending_clear ? pending_data : pending_old_value + pending_data;
    assign port_a_write = op_pending || seed_valid;
    assign port_a_addr = op_pending ? pending_addr : seed_addr;
    assign port_a_data = op_pending ? pending_result : seed_data;
    assign read_data = read_data_r;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            op_pending <= 1'b0;
            pending_clear <= 1'b0;
            pending_addr <= '0;
            pending_data <= '0;
            pending_old_value <= '0;
            op_result <= '0;
            read_pending_pipe <= '0;
            read_addr_r <= '0;
            read_mem_data <= '0;
            read_valid <= 1'b0;
            read_data_r <= '0;
        end else begin
            // Vivado can split this true dual-port 256x32 RAM across cascaded
            // RAMB18s and absorb different output registers into each half.
            // Keep the requested address stable and sample the port on every
            // cycle; only advertise it after four settling clocks. This gives
            // simulation and every legal RAMB18 mapping one fixed contract.
            read_addr_r <= read_addr;
            read_mem_data <= mem[read_addr_r];
            read_data_r <= read_mem_data;
            read_pending_pipe <= {read_pending_pipe[2:0], read_req};
            read_valid <= read_pending_pipe[3];

            if (port_a_write) begin
                mem[port_a_addr] <= port_a_data;
            end

            if (op_pending) begin
                op_result <= pending_result;
                op_pending <= 1'b0;
            end

            if (op_valid && op_ready) begin
                pending_clear <= op_clear;
                pending_addr <= op_addr;
                pending_data <= op_data;
                pending_old_value <= mem[op_addr];
                op_pending <= 1'b1;
            end
        end
    end

endmodule
