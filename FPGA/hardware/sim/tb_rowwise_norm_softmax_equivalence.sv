module tb_rowwise_norm_softmax_equivalence;
    localparam int N = 197;
    logic clk=0, rst_n=0, start_shared=0, start_ln=0, start_sm=0;
    logic mode=0, input_valid=0, param_valid=1, output_ready=1;
    logic [7:0] vector_length;
    logic [31:0] epsilon_term=32'd17;
    logic signed [7:0] input_data;
    logic signed [15:0] gamma_q8=16'sd173, beta_q8=-16'sd61;
    logic shared_in_ready, shared_param_ready, shared_out_valid, shared_last;
    logic shared_busy, shared_done, shared_error;
    logic signed [7:0] shared_out;
    logic ln_in_ready, ln_param_ready, ln_out_valid, ln_last, ln_busy, ln_done, ln_error;
    logic signed [7:0] ln_out;
    logic sm_in_ready, sm_out_valid, sm_last, sm_busy, sm_done, sm_error;
    logic [7:0] sm_out;
    logic signed [7:0] values[0:N-1], reference[0:N-1], candidate[0:N-1];
    integer ref_count=0, candidate_count=0, errors=0;

    always #5 clk=~clk;

    rowwise_norm_softmax_engine #(.MAX_LENGTH(N)) shared (
        .clk, .rst_n, .start(start_shared), .softmax_mode(mode), .vector_length,
        .epsilon_term, .input_valid, .input_ready(shared_in_ready), .input_data,
        .param_valid, .param_ready(shared_param_ready), .gamma_q8, .beta_q8,
        .output_valid(shared_out_valid), .output_ready, .output_data(shared_out),
        .output_last(shared_last), .busy(shared_busy), .done(shared_done), .error(shared_error)
    );
    layernorm_int8 #(.MAX_LENGTH(N)) legacy_ln (
        .clk, .rst_n, .start(start_ln), .vector_length, .epsilon_term,
        .input_valid(input_valid && !mode), .input_ready(ln_in_ready), .input_data,
        .param_valid, .param_ready(ln_param_ready), .gamma_q8, .beta_q8,
        .output_valid(ln_out_valid), .output_ready, .output_data(ln_out),
        .output_last(ln_last), .busy(ln_busy), .done(ln_done), .error(ln_error)
    );
    softmax_int8 #(.MAX_LENGTH(N)) legacy_sm (
        .clk, .rst_n, .start(start_sm), .vector_length,
        .input_valid(input_valid && mode), .input_ready(sm_in_ready), .input_data,
        .output_valid(sm_out_valid), .output_ready,
        .output_probability_q7(sm_out), .output_last(sm_last),
        .busy(sm_busy), .done(sm_done), .error(sm_error)
    );

    always_ff @(posedge clk) begin
        if (shared_out_valid) begin
            candidate[candidate_count] <= shared_out;
            candidate_count <= candidate_count + 1;
        end
        if (!mode && ln_out_valid) begin
            reference[ref_count] <= ln_out;
            ref_count <= ref_count + 1;
        end
        if (mode && sm_out_valid) begin
            reference[ref_count] <= sm_out;
            ref_count <= ref_count + 1;
        end
    end

    task automatic run_case(input logic softmax, input integer length); begin
        mode=softmax; vector_length=length; ref_count=0; candidate_count=0;
        @(negedge clk); start_shared=1; start_ln=!softmax; start_sm=softmax;
        @(negedge clk); start_shared=0; start_ln=0; start_sm=0;
        for (int i=0;i<length;i++) begin
            wait(shared_in_ready && (softmax ? sm_in_ready : ln_in_ready));
            input_data=values[i]; input_valid=1;
            @(posedge clk); @(negedge clk);
        end
        input_valid=0;
        wait(ref_count==length && candidate_count==length); @(negedge clk);
        for (int i=0;i<length;i++) if (candidate[i] !== reference[i]) begin
            $error("mode=%0d index=%0d legacy=%0d shared=%0d",softmax,i,reference[i],candidate[i]);
            errors++;
        end
        if (shared_error || (softmax ? sm_error : ln_error)) begin
            $error("engine error mode=%0d",softmax); errors++;
        end
    end endtask

    initial begin
        for (int i=0;i<N;i++) values[i]=$signed(((i*73+19)%256)-128);
        repeat(3) @(posedge clk); rst_n=1;
        run_case(0,192);
        gamma_q8=256; beta_q8=0;
        run_case(1,197);
        if (errors) $fatal(1,"shared/legacy equivalence failed errors=%0d",errors);
        $display("Shared LayerNorm/Softmax legacy-equivalence golden trace passed");
        $finish;
    end
    initial begin #5000000; $fatal(1,"equivalence timeout"); end
endmodule
