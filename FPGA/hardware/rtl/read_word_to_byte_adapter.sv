module read_word_to_byte_adapter #(
    parameter int DEPTH = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        word_valid,
    output logic        word_ready,
    input  logic [31:0] word_data,
    input  logic [3:0]  word_keep,
    input  logic        word_last,
    output logic        byte_valid,
    input  logic        byte_ready,
    output logic [7:0]  byte_data,
    output logic        byte_last,
    output logic        busy
);
    localparam int PTR_WIDTH = DEPTH <= 2 ? 1 : $clog2(DEPTH);
    localparam int COUNT_WIDTH = $clog2(DEPTH + 1);

    logic [31:0] data_fifo [0:DEPTH-1];
    logic [3:0] keep_fifo [0:DEPTH-1];
    logic last_fifo [0:DEPTH-1];
    logic [PTR_WIDTH-1:0] write_ptr, read_ptr;
    logic [COUNT_WIDTH-1:0] count;
    logic [1:0] byte_index;
    logic [1:0] last_lane;

    always_comb begin
        if (keep_fifo[read_ptr][3]) last_lane = 2'd3;
        else if (keep_fifo[read_ptr][2]) last_lane = 2'd2;
        else if (keep_fifo[read_ptr][1]) last_lane = 2'd1;
        else last_lane = 2'd0;
    end

    wire pop_word = byte_valid && byte_ready && byte_index == last_lane;
    wire push_word = word_valid && word_ready;

    assign word_ready = count < DEPTH || pop_word;
    assign byte_valid = count != 0 && keep_fifo[read_ptr][byte_index];
    assign byte_data = data_fifo[read_ptr][byte_index*8 +: 8];
    assign byte_last = byte_valid && last_fifo[read_ptr] && byte_index == last_lane;
    assign busy = count != 0;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            write_ptr <= '0;
            read_ptr <= '0;
            count <= '0;
            byte_index <= '0;
        end else begin
            if (push_word) begin
                data_fifo[write_ptr] <= word_data;
                keep_fifo[write_ptr] <= word_keep;
                last_fifo[write_ptr] <= word_last;
                write_ptr <= write_ptr == DEPTH-1 ? '0 : write_ptr + 1'b1;
            end
            if (byte_valid && byte_ready) begin
                if (byte_index == last_lane) begin
                    byte_index <= '0;
                    read_ptr <= read_ptr == DEPTH-1 ? '0 : read_ptr + 1'b1;
                end else begin
                    byte_index <= byte_index + 1'b1;
                end
            end
            unique case ({push_word, pop_word})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: begin end
            endcase
        end
    end
endmodule
