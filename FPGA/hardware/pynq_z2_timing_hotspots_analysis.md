# PYNQ-Z2 Post-Route Timing Hotspots

Analysis source: the routed checkpoint used for the deployed 100 MHz image.
The design meets timing with `+0.008 ns` WNS, so it has effectively no setup
margin for placement variation or a higher clock target.

| Rank/family | Worst slack | Datapath | Logic/route | Finding |
| --- | ---: | ---: | --- | --- |
| AveragePool plane address | +0.008 ns | 9.842 ns | 4.231/5.611 ns | 14 levels: eight CARRY4s plus six LUTs |
| TileLoader prepared length | +0.057 ns | 9.681 ns | 5.140/4.541 ns | 15-level address/length decision cone |
| Systolic-array accumulator CE | +0.091 ns | 8.987 ns | 0.419/8.568 ns | 95.3% routing; this is control distribution, not MAC arithmetic |
| DDR cache write | +0.094 ns | 9.562 ns | 1.507/8.055 ns | 84.2% routing from HP0 to distributed cache storage |
| Conv weight-tile stride | +0.191 ns | 9.705 ns | carry/LUT arithmetic | Secondary control/address cone |

## Primary Cause

`avgpool_ddr.sv` computes `addr_plane_r <= ni * c_r` in `ADDR_PLANE_MUL`.
This generic 16x16 multiply is repeated for every valid pooling sample and is
implemented as LUT/carry arithmetic. Its upper result bits form all three worst
independent setup endpoints.

The best correction is to compute or update the `(ni * channels + ci)` plane
base once when the output plane changes, then reuse it for all pooling samples.
If generic multiplication must remain, it should be decomposed and registered
like the later row and linear address products. Incremental plane/row pointers
are preferable because they improve both timing and pooling cycle count.

## Secondary Causes

1. Pipeline the TileLoader prepared-length/boundary decision into separate
   remaining-span and clamp stages.
2. Replicate or localize the systolic accumulator-enable control. Its delay is
   almost entirely routing across the array.
3. Bank or floorplan the DDR read cache near HP0 and register the cache-write
   fanout. Its critical paths are also route dominated.

The AveragePool address recurrence should be fixed first. It is the worst path,
it consumes runtime cycles, and its correction does not alter the MAC datapath.
