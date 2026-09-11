"""Read the PYNQ FTDI UART directly when Windows has not created a COM port."""
from __future__ import annotations

import argparse
import ctypes
import time


FT_OK = 0
FT_OPEN_BY_SERIAL_NUMBER = 1


def check(status: int, operation: str) -> None:
    if status != FT_OK:
        raise RuntimeError(f"{operation} failed with FTDI status {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="1234-tulB")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--command", help="Console command to send after opening")
    args = parser.parse_args()

    dll = ctypes.WinDLL(r"C:\Windows\System32\ftd2xx.dll")
    handle = ctypes.c_void_p()
    dll.FT_OpenEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                              ctypes.POINTER(ctypes.c_void_p)]
    dll.FT_OpenEx.restype = ctypes.c_ulong
    check(dll.FT_OpenEx(ctypes.c_char_p(args.serial.encode("ascii")),
                        FT_OPEN_BY_SERIAL_NUMBER, ctypes.byref(handle)), "FT_OpenEx")
    try:
        check(dll.FT_SetBaudRate(handle, args.baud), "FT_SetBaudRate")
        check(dll.FT_SetDataCharacteristics(handle, 8, 0, 0),
              "FT_SetDataCharacteristics")
        check(dll.FT_SetFlowControl(handle, 0, 0, 0), "FT_SetFlowControl")
        check(dll.FT_SetTimeouts(handle, 200, 200), "FT_SetTimeouts")
        if args.command is not None or args.prompt:
            encoded = ((args.command or "") + "\r\n").encode("utf-8")
            payload = ctypes.create_string_buffer(encoded)
            written = ctypes.c_ulong()
            check(dll.FT_Write(handle, payload, len(encoded), ctypes.byref(written)),
                  "FT_Write")

        deadline = time.monotonic() + args.seconds
        captured = bytearray()
        while time.monotonic() < deadline:
            queued = ctypes.c_ulong()
            check(dll.FT_GetQueueStatus(handle, ctypes.byref(queued)),
                  "FT_GetQueueStatus")
            if queued.value:
                buffer = ctypes.create_string_buffer(queued.value)
                received = ctypes.c_ulong()
                check(dll.FT_Read(handle, buffer, queued.value,
                                  ctypes.byref(received)), "FT_Read")
                captured.extend(buffer.raw[:received.value])
            else:
                time.sleep(0.05)
        print(captured.decode("utf-8", errors="replace"), end="")
        if not captured:
            print("<no UART data received>")
    finally:
        dll.FT_Close(handle)


if __name__ == "__main__":
    main()
