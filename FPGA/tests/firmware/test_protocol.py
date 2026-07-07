import ctypes
import unittest

from common import (
    BC_CMD_ALLOC,
    BC_STATUS_INVALID_COMMAND,
    BC_STATUS_OK,
    BcudaHeader,
    BcudaResponseHeader,
    compile_firmware_library,
)


class FirmwareProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("protocol")
        cls.lib.bcuda_validate_header.argtypes = [ctypes.POINTER(BcudaHeader), ctypes.c_uint32]
        cls.lib.bcuda_validate_header.restype = ctypes.c_int
        cls.lib.bcuda_init_response.argtypes = [
            ctypes.POINTER(BcudaResponseHeader),
            ctypes.POINTER(BcudaHeader),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]

    def test_bcuda_validate_header_accepts_valid_header(self):
        header = BcudaHeader(0x42435544, 1, BC_CMD_ALLOC, 77, ctypes.sizeof(BcudaHeader), 0, 0, 0)
        self.assertEqual(self.lib.bcuda_validate_header(ctypes.byref(header), BC_CMD_ALLOC), BC_STATUS_OK)

    def test_bcuda_validate_header_rejects_bad_magic(self):
        header = BcudaHeader(0, 1, BC_CMD_ALLOC, 77, ctypes.sizeof(BcudaHeader), 0, 0, 0)
        self.assertEqual(self.lib.bcuda_validate_header(ctypes.byref(header), BC_CMD_ALLOC), BC_STATUS_INVALID_COMMAND)

    def test_bcuda_init_response_copies_request_fields(self):
        request = BcudaHeader(0x42435544, 1, BC_CMD_ALLOC, 99, ctypes.sizeof(BcudaHeader), 0, 0, 0)
        response = BcudaResponseHeader()
        self.lib.bcuda_init_response(ctypes.byref(response), ctypes.byref(request), BC_STATUS_OK, ctypes.sizeof(response), 0)
        self.assertEqual(response.magic, 0x42435544)
        self.assertEqual(response.command, BC_CMD_ALLOC)
        self.assertEqual(response.request_id, 99)
        self.assertEqual(response.status, BC_STATUS_OK)


if __name__ == "__main__":
    unittest.main()
