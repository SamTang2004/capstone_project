import ctypes
import unittest

from common import (
    BC_STATUS_GENERAL_ERROR,
    BC_STATUS_INVALID_HANDLE,
    BC_STATUS_OK,
    BcudaBuffer,
    alloc_buffer,
    compile_firmware_library,
    init_buffer_manager,
    tensor_desc,
)


class FirmwareBufferManagerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("buffer")
        cls.lib.bcuda_alloc_buffer.restype = ctypes.c_int
        cls.lib.bcuda_free_buffer.restype = ctypes.c_int
        cls.lib.bcuda_get_buffer.restype = ctypes.c_int
        cls.lib.bcuda_write_buffer.restype = ctypes.c_int
        cls.lib.bcuda_read_buffer.restype = ctypes.c_int
        cls.lib.bcuda_buffer_bytes_used.restype = ctypes.c_uint64

    def setUp(self):
        self.arena = init_buffer_manager(self.lib)

    def test_bcuda_alloc_buffer_returns_handle_and_metadata(self):
        buffer_id = alloc_buffer(self.lib, 16, tensor_desc(dim0=16))
        out = ctypes.POINTER(BcudaBuffer)()
        self.assertEqual(self.lib.bcuda_get_buffer(buffer_id, ctypes.byref(out)), BC_STATUS_OK)
        self.assertEqual(out.contents.buffer_id, buffer_id)
        self.assertEqual(out.contents.size_bytes, 16)
        self.assertEqual(out.contents.shape[0], 16)

    def test_bcuda_write_buffer_and_read_buffer_round_trip(self):
        buffer_id = alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        src = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        dst = (ctypes.c_uint8 * 4)()
        self.assertEqual(self.lib.bcuda_write_buffer(buffer_id, 0, src, 4), BC_STATUS_OK)
        self.assertEqual(self.lib.bcuda_read_buffer(buffer_id, 0, dst, 4), BC_STATUS_OK)
        self.assertEqual(list(dst), [1, 2, 3, 4])

    def test_bcuda_write_buffer_rejects_out_of_range_write(self):
        buffer_id = alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        src = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        self.assertEqual(self.lib.bcuda_write_buffer(buffer_id, 2, src, 4), BC_STATUS_GENERAL_ERROR)

    def test_bcuda_free_buffer_invalidates_handle(self):
        buffer_id = alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        self.assertEqual(self.lib.bcuda_free_buffer(buffer_id), BC_STATUS_OK)
        out = ctypes.POINTER(BcudaBuffer)()
        self.assertEqual(self.lib.bcuda_get_buffer(buffer_id, ctypes.byref(out)), BC_STATUS_INVALID_HANDLE)

    def test_bcuda_buffer_bytes_used_tracks_arena_offset(self):
        alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        self.assertGreaterEqual(self.lib.bcuda_buffer_bytes_used(), 4)


if __name__ == "__main__":
    unittest.main()
