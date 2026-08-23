set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set bit_file [file join $hw_dir "bcuda_pynq_z2.bit"]

if {![file exists $bit_file]} {
    error "Missing bitstream: $bit_file"
}

source [file join $script_dir "fpga_manager_bin.tcl"]
export_fpga_manager_bin $script_dir $bit_file [file join $hw_dir "bcuda_pynq_z2.bin"]
puts "Exported PYNQ-Z2 FPGA-manager image: [file join $hw_dir bcuda_pynq_z2.bin]"
