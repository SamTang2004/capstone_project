"""Minimal FTDI D2XX UART console for PYNQ-Z2 recovery on Windows."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time


FT_OPEN_BY_DESCRIPTION = 2
FT_PURGE_RX = 1
FT_PURGE_TX = 2


def check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with FT_STATUS={status}")


def devices(dll: ctypes.WinDLL) -> list[tuple[str, str]]:
    count = ctypes.c_ulong()
    check(dll.FT_CreateDeviceInfoList(ctypes.byref(count)), "FT_CreateDeviceInfoList")
    found = []
    for index in range(count.value):
        flags = ctypes.c_ulong()
        kind = ctypes.c_ulong()
        device_id = ctypes.c_ulong()
        location = ctypes.c_ulong()
        serial = ctypes.create_string_buffer(16)
        description = ctypes.create_string_buffer(64)
        handle = ctypes.c_void_p()
        check(
            dll.FT_GetDeviceInfoDetail(
                index,
                ctypes.byref(flags),
                ctypes.byref(kind),
                ctypes.byref(device_id),
                ctypes.byref(location),
                serial,
                description,
                ctypes.byref(handle),
            ),
            "FT_GetDeviceInfoDetail",
        )
        found.append((serial.value.decode(errors="replace"), description.value.decode(errors="replace")))
    return found


def open_uart(dll: ctypes.WinDLL, description_text: str, deadline: float) -> ctypes.c_void_p:
    description = ctypes.create_string_buffer(description_text.encode())
    while time.monotonic() < deadline:
        handle = ctypes.c_void_p()
        status = dll.FT_OpenEx(description, FT_OPEN_BY_DESCRIPTION, ctypes.byref(handle))
        if status == 0:
            check(dll.FT_SetBaudRate(handle, 115200), "FT_SetBaudRate")
            check(dll.FT_SetDataCharacteristics(handle, 8, 0, 0), "FT_SetDataCharacteristics")
            check(dll.FT_SetFlowControl(handle, 0, 0, 0), "FT_SetFlowControl")
            check(dll.FT_SetTimeouts(handle, 100, 100), "FT_SetTimeouts")
            return handle
        time.sleep(0.01)
    raise RuntimeError(f"timed out opening FTDI UART {description_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", default="USB Serial Converter B")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--send", default="\r")
    parser.add_argument("--purge", action="store_true")
    parser.add_argument("--interrupt-boot", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    dll = ctypes.WinDLL("ftd2xx.dll")
    available = devices(dll)
    if args.list:
        for serial, description in available:
            print(f"{serial}: {description}")
        return
    if not any(description == args.description for _, description in available):
        raise RuntimeError(f"FTDI device {args.description!r} not found; available={available}")

    deadline = time.monotonic() + args.seconds
    handle = open_uart(dll, args.description, deadline)
    try:
        if args.purge:
            check(dll.FT_Purge(handle, FT_PURGE_RX | FT_PURGE_TX), "FT_Purge")
        if args.send:
            payload = args.send.encode().decode("unicode_escape").encode()
            written = ctypes.c_ulong()
            check(dll.FT_Write(handle, payload, len(payload), ctypes.byref(written)), "FT_Write")
        last_interrupt = 0.0
        while time.monotonic() < deadline:
            queued = ctypes.c_ulong()
            status = dll.FT_GetQueueStatus(handle, ctypes.byref(queued))
            if status != 0:
                dll.FT_Close(handle)
                handle = open_uart(dll, args.description, deadline)
                continue
            if args.interrupt_boot and time.monotonic() - last_interrupt >= 0.05:
                interrupt = b" "
                written = ctypes.c_ulong()
                status = dll.FT_Write(handle, interrupt, len(interrupt), ctypes.byref(written))
                if status != 0:
                    continue
                last_interrupt = time.monotonic()
            if queued.value:
                chunk = ctypes.create_string_buffer(min(queued.value, 4096))
                received = ctypes.c_ulong()
                check(dll.FT_Read(handle, chunk, len(chunk), ctypes.byref(received)), "FT_Read")
                sys.stdout.buffer.write(chunk.raw[: received.value])
                sys.stdout.buffer.flush()
            else:
                time.sleep(0.02)
    finally:
        dll.FT_Close(handle)


if __name__ == "__main__":
    main()
