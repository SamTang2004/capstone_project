"""Convert a Vivado .bit file into the raw payload expected by fpga_manager."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def bit_to_fpga_manager_payload(contents: bytes) -> bytes:
    """Match PYNQ's bit2bin conversion without requiring the PYNQ package."""
    if len(contents) < 4:
        raise ValueError("bitstream is too short")

    first_field_size = struct.unpack(">H", contents[:2])[0]
    offset = 2 + first_field_size + 2
    while offset < len(contents):
        field = contents[offset]
        offset += 1
        if field == 0x65:
            if offset + 4 > len(contents):
                raise ValueError("truncated bitstream payload length")
            payload_size = struct.unpack(">I", contents[offset : offset + 4])[0]
            offset += 4
            payload = contents[offset : offset + payload_size]
            if len(payload) != payload_size or offset + payload_size != len(contents):
                raise ValueError("invalid bitstream payload length")
            if payload_size % 4:
                raise ValueError("bitstream payload is not word aligned")
            return b"".join(payload[index : index + 4][::-1] for index in range(0, payload_size, 4))

        if offset + 2 > len(contents):
            raise ValueError("truncated bitstream header")
        field_size = struct.unpack(">H", contents[offset : offset + 2])[0]
        offset += 2 + field_size

    raise ValueError("bitstream payload field was not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bit_file", type=Path)
    parser.add_argument("bin_file", type=Path)
    args = parser.parse_args()
    args.bin_file.write_bytes(bit_to_fpga_manager_payload(args.bit_file.read_bytes()))


if __name__ == "__main__":
    main()
