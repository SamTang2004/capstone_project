module gemm_array_cluster_8 #(
    parameter int ARRAY_COUNT = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    load_valid,
    input  logic                    load_matrix,
    input  logic [2:0]              load_array,
    input  logic [7:0]              load_addr,
    input  logic [7:0]              load_data,
    input  logic                    load_word_valid,
    input  logic [7:0]              load_word_addr,
    input  logic [7:0]              load_word_stride,
    input  logic [63:0]             load_word_data,
    input  logic [7:0]              load_word_keep,
    input  logic                    invalidate_valid,
    input  logic                    invalidate_matrix,
    input  logic                    start,
    input  logic [3:0]              active_arrays,
    output logic                    busy,
    output logic                    done,
    input  logic [2:0]              result_array,
    input  logic [7:0]              result_addr,
    output logic signed [31:0]      result_data
);
    logic [ARRAY_COUNT-1:0] core_busy;
    logic [ARRAY_COUNT-1:0] core_done;
    logic signed [31:0] core_result [0:ARRAY_COUNT-1];

    genvar i;
    generate
        for (i = 0; i < ARRAY_COUNT; i = i + 1) begin : cores
            gemm_16x16_core core (
                .clk(clk),
                .rst_n(rst_n),
                .load_valid(load_valid && (!load_matrix || load_array == i[2:0])),
                .load_matrix(load_matrix),
                .load_addr(load_addr),
                .load_data(load_data),
                .load_word_valid(load_word_valid && (!load_matrix || load_array == i[2:0])),
                .load_word_addr(load_word_addr),
                .load_word_stride(load_word_stride),
                .load_word_data(load_word_data),
                .load_word_keep(load_word_keep),
                .invalidate_valid(invalidate_valid && (!invalidate_matrix || i < active_arrays)),
                .invalidate_matrix(invalidate_matrix),
                .start(start && i < active_arrays),
                .busy(core_busy[i]),
                .done(core_done[i]),
                .result_addr(result_addr),
                .result_data(core_result[i])
            );
        end
    endgenerate

    always_comb begin
        busy = 1'b0;
        done = 1'b1;
        for (int j = 0; j < ARRAY_COUNT; j++) begin
            if (j < active_arrays) begin
                busy = busy || core_busy[j];
                done = done && core_done[j];
            end
        end
        if (active_arrays == 0) begin
            done = 1'b0;
        end
    end

    assign result_data = core_result[result_array];
endmodule
