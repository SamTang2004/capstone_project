module w4a8_operand_buffers #(
    parameter int DEPTH = 1024,
    parameter int LENGTH_WIDTH = $clog2(DEPTH + 1)
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    fill_start,
    output logic                    fill_start_ready,
    input  logic [LENGTH_WIDTH-1:0] k_steps,

    input  logic                    activation_write_valid,
    output logic                    activation_write_ready,
    input  logic [63:0]             activation_write_data,
    input  logic                    weight_write_valid,
    output logic                    weight_write_ready,
    input  logic [31:0]             weight_write_data,
    output logic                    fill_busy,
    output logic                    fill_done,

    input  logic                    drain_start,
    output logic                    drain_start_ready,
    output logic                    tile_available,
    output logic                    operand_valid,
    input  logic                    operand_ready,
    output logic [63:0]             activation_vector,
    output logic [31:0]             packed_weight_vector,
    output logic                    operand_last,
    output logic                    drain_busy,
    output logic                    drain_done,
    output logic                    error
);
    logic activation_start_ready, weight_start_ready;
    logic activation_write_busy, weight_write_busy;
    logic activation_write_done, weight_write_done;
    logic activation_read_start_ready, weight_read_start_ready;
    logic activation_available, weight_available;
    logic activation_valid, weight_valid;
    logic activation_ready, weight_ready;
    logic activation_last, weight_last;
    logic activation_read_busy, weight_read_busy;
    logic activation_read_done, weight_read_done;
    logic activation_error, weight_error, alignment_error;
    logic common_fill_start, common_drain_start;
    logic activation_fill_complete, weight_fill_complete;

    assign fill_start_ready = activation_start_ready && weight_start_ready;
    assign common_fill_start = fill_start && fill_start_ready;
    assign fill_busy = activation_write_busy || weight_write_busy;

    assign tile_available = activation_available && weight_available;
    assign drain_start_ready = activation_read_start_ready && weight_read_start_ready;
    assign common_drain_start = drain_start && drain_start_ready;
    assign drain_busy = activation_read_busy || weight_read_busy;
    assign drain_done = activation_read_done && weight_read_done;

    assign operand_valid = activation_valid && weight_valid;
    assign activation_ready = operand_ready && weight_valid;
    assign weight_ready = operand_ready && activation_valid;
    assign operand_last = activation_last && weight_last;
    assign error = activation_error || weight_error || alignment_error;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            alignment_error <= 1'b0;
            activation_fill_complete <= 1'b0;
            weight_fill_complete <= 1'b0;
            fill_done <= 1'b0;
        end else begin
            fill_done <= 1'b0;
            if (operand_valid && activation_last != weight_last)
                alignment_error <= 1'b1;

            if (common_fill_start) begin
                activation_fill_complete <= 1'b0;
                weight_fill_complete <= 1'b0;
            end else begin
                if (activation_write_done) activation_fill_complete <= 1'b1;
                if (weight_write_done) weight_fill_complete <= 1'b1;
                if ((activation_write_done && (weight_write_done || weight_fill_complete)) ||
                    (weight_write_done && activation_fill_complete)) begin
                    fill_done <= 1'b1;
                    activation_fill_complete <= 1'b0;
                    weight_fill_complete <= 1'b0;
                end
            end
        end
    end

    ping_pong_tile_buffer #(
        .DATA_WIDTH(64),
        .DEPTH(DEPTH)
    ) activation_buffer (
        .clk(clk), .rst_n(rst_n),
        .write_start(common_fill_start), .write_start_ready(activation_start_ready),
        .write_length(k_steps), .write_valid(activation_write_valid),
        .write_ready(activation_write_ready), .write_data(activation_write_data),
        .write_busy(activation_write_busy), .write_done(activation_write_done),
        .read_start(common_drain_start), .read_start_ready(activation_read_start_ready),
        .read_available(activation_available), .read_valid(activation_valid),
        .read_ready(activation_ready), .read_data(activation_vector),
        .read_last(activation_last), .read_busy(activation_read_busy),
        .read_done(activation_read_done), .error(activation_error)
    );

    ping_pong_tile_buffer #(
        .DATA_WIDTH(32),
        .DEPTH(DEPTH)
    ) weight_buffer (
        .clk(clk), .rst_n(rst_n),
        .write_start(common_fill_start), .write_start_ready(weight_start_ready),
        .write_length(k_steps), .write_valid(weight_write_valid),
        .write_ready(weight_write_ready), .write_data(weight_write_data),
        .write_busy(weight_write_busy), .write_done(weight_write_done),
        .read_start(common_drain_start), .read_start_ready(weight_read_start_ready),
        .read_available(weight_available), .read_valid(weight_valid),
        .read_ready(weight_ready), .read_data(packed_weight_vector),
        .read_last(weight_last), .read_busy(weight_read_busy),
        .read_done(weight_read_done), .error(weight_error)
    );
endmodule
