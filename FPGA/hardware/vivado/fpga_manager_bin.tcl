proc export_fpga_manager_bin {script_dir bit_file bin_file} {
    # Vivado exports PYTHONHOME/PYTHONPATH for its bundled Python 3.8. Those
    # variables poison a separately installed Python when Tcl launches it.
    set had_home [info exists ::env(PYTHONHOME)]
    set had_path [info exists ::env(PYTHONPATH)]
    if {$had_home} { set saved_home $::env(PYTHONHOME); unset ::env(PYTHONHOME) }
    if {$had_path} { set saved_path $::env(PYTHONPATH); unset ::env(PYTHONPATH) }
    set failed [catch {
        exec python [file join $script_dir "pynq_bit2bin.py"] $bit_file $bin_file
    } message options]
    if {$had_home} { set ::env(PYTHONHOME) $saved_home }
    if {$had_path} { set ::env(PYTHONPATH) $saved_path }
    if {$failed} { return -options $options $message }
}
