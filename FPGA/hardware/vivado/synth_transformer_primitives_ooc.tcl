set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set report_dir [file join $hw_dir "reports" "transformer_primitives"]
file mkdir $report_dir

set modules {
    transformer_debug_counters transformer_debug_counters.sv
    residual_add_int8 residual_add_int8.sv
    bias_add_int32 bias_add_int32.sv
    gelu_int8 gelu_int8.sv
    layernorm_int8 layernorm_int8.sv
    softmax_int8 softmax_int8.sv
}

foreach {top source} $modules {
    create_project -in_memory -part xc7z020clg400-1
    read_verilog -sv [file join $hw_dir "rtl" $source]
    synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
    report_utilization -file [file join $report_dir "${top}_utilization.rpt"]
    report_timing_summary -file [file join $report_dir "${top}_timing.rpt"]
    puts "SYNTHESIZED $top"
    close_project
}
