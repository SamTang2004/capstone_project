set script_dir [file dirname [file normalize [info script]]]
set hw_dir [file normalize [file join $script_dir ".."]]
set bit_file [file join $hw_dir "bcuda_pynq_z2.bit"]

if {![file exists $bit_file]} {
    error "Missing bitstream: $bit_file"
}

write_cfgmem -force -format bin -interface smapx32 -size 16 \
    -loadbit "up 0x0 $bit_file" \
    -file [file join $hw_dir "bcuda_pynq_z2.bin"]
puts "Exported PYNQ-Z2 FPGA-manager image: [file join $hw_dir bcuda_pynq_z2.bin]"
