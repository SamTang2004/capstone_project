# First manual T0 board attempt

Endpoint: 192.168.2.99:9000. The scheduled fpga-t0 heartbeat was paused.

The existing service-based transformer graph smoke passed W8A8 and W4A8 MLP
and three-head attention golden checks. Summed GEMM windows were 3563/27049
cycles for W8 MLP/attention and 3518/27056 for W4. These are graph operation
windows, not single-tile latency measurements.

The new independent sweep first submitted W4 M=1, N=8, K=16 with tiled INT8
activations, packed W4 weights and row-major INT32 output. EXECUTE_GRAPH timed
out before the first result. Subsequent cleanup requests also timed out; buffer
release could not be confirmed. No successful sweep samples were obtained.

After the timeout, two ICMP probes received no reply and SSH port 22 connection
attempts timed out. Earlier SSH key authentication had reached the host but was
rejected. The user then supplied authorization to use the PYNQ default password;
the subsequent password-mode SSH connection timed out before authentication.

Actual FCLK and loaded firmware/bitstream hashes remain unverified. The 100 MHz
diagnostic setting was only a host assumption. No device reset, reprogramming,
or service restart was performed. The temporal association with the W4 GEMV
request does not establish whether the failure is in PL, firmware, or network.

Next: restore/confirm connectivity, inspect daemon and kernel logs via SSH or
serial, then isolate the raw INT32 GEMM path versus the proven fused graph path.
Do not repeat the full sweep before checking that failure.

## UART recovery checks

Followed `FPGA/DEPLOYMENT_FROM_GROUND_ZERO.md` section 11. D2XX enumerated
`1234-tulA / TUL A` and `1234-tulB / TUL B`. UART B opened at 115200 baud,
but CR, Ctrl-C followed by CR, and the guide's read-only service query produced
no received UART data. The prompt and service query were also tested outside
the shell sandbox; neither returned data.

Read-only XSDB target enumeration succeeded: APU, both Cortex-A9 cores marked
`Running`, and `xc7z020`. This establishes JTAG visibility, not Linux health.
No reset, halt, bitstream programming, or service restart was issued. Recovery
is still blocked on an unresponsive Linux console/network; a system reset is
a separate next step with potential loss of unflushed SD-card writes.

## Authorized JTAG recovery and connection restored

After user authorization, XSDB `rst -system` on the enumerated APU restored
UART boot output. A name-filter attempt failed before resetting; target 1
was then selected from the observed target list. Linux boot was slow. A
second reset was queued prematurely before the first boot's delayed output
was inspected; it interrupted a boot that had actually reached login. The
second boot was stopped in U-Boot, then resumed with `boot`, without changing
or saving environment variables. Two system resets were executed in total.

U-Boot reported an invalid environment and no valid MAC; Linux assigned a
random MAC. Kernel/FDT hashes verified successfully. EXT4 journal recovery
completed, and systemd replaced an unclean/corrupted journal file. Linux
6.6.10 booted into the existing PYNQ Ubuntu 22.04 installation. Board wall
clock was May 12, 2025, so board log timestamps are not host dates.

UART autologin and password-authenticated SSH both succeeded. Board eth0 is
192.168.2.99/24, host Ethernet is 192.168.2.1/24. The existing enabled FPGA
and daemon services started automatically after bootpy; both became active,
FPGA state became `operating`, and TCP 9000 listened. Do not mistake their
initial inactive state during boot for missing deployment.

Installed SHA256 baseline (read on board):

- server: `231736b1f9b286d926171a3e31625988c59d1b2305e2c6fb6603ff6300fc7f97`
- bit: `5863b5c1f7c87ac9e1c88087eda47c73bab5e426e2918e0de06427aafb92d0da`
- hwh: `2fbb3ef754e29a510783846339e80b9d60415e62e23b3ad65a64cc966e2527ec`

Setup logs report 100 MHz and FCLK0 register `0x00200500`. The old sysfs
`/sys/class/fclk/fclk0/rate` path does not exist on this image; measured clock
rate is not yet independently verified.

Post-reboot original transformer smoke returned status 7 at W8A8 MLP layer 0
(LAYERNORM), before testing W4A8. Connectivity stayed responsive. The previous
micro-GEMV sweep was not repeated. Applied the documented status-7 recovery:
stop daemon, restart FPGA setup using the same installed overlay, then start
daemon. No new binaries or bitstreams were deployed.

Both services returned active after that recovery, but one repeat of the
original smoke still failed at the same W8A8 LayerNorm with status 7. Thus
UART/SSH recovery is complete, but accelerator functional readiness is not.
Do not proceed to T0 measurements until this baseline failure is isolated.
