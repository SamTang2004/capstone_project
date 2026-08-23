import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device
from budgetcuda.compiler import (HostBuffer, Layer, ModelPackage, ResidentBufferPool,
                                 build_tiny_conv_relu_package, upload_model)
from budgetcuda.dtypes import BufferUsage, DType, OpType
from budgetcuda.model import Model


class RecordingDevice(Device):
    def __init__(self):
        self.writes = []
        self.graph_calls = []
        self.output = [1, 2, 3]

    def write(self, tensor, np_array, offset_bytes=0):
        self.writes.append((tensor, np_array, offset_bytes))

    def execute_graph(self, descriptor_tensor, input_tensor, output_tensor, flags=0):
        self.graph_calls.append((descriptor_tensor, input_tensor, output_tensor, flags))

    def read(self, tensor, offset_bytes=0, nbytes=None):
        return self.output

    def free(self, tensor):
        self.freed.append(tensor)


class ModelTest(unittest.TestCase):
    def test_external_resident_weight_is_borrowed_by_model(self):
        dev = Device.mock()
        package = ModelPackage(
            buffers=[
                HostBuffer("input", (1, 2), DType.INT8),
                HostBuffer.from_values("weight", (2, 1), DType.INT8, [3, 4]),
                HostBuffer("output", (1, 1), DType.INT32, usage=BufferUsage.OUTPUT),
            ],
            layers=[Layer(OpType.GEMM, input0=0, weight=1, output=2, n=1, c=2, out_c=1)],
            input_index=0,
            output_index=2,
        )
        pool = ResidentBufferPool(dev)
        pool.upload([package.buffers[1]])
        resident_id = pool.tensors["weight"].buffer_id
        model = upload_model(dev, package, external_tensors=pool.tensors)
        self.assertEqual(values := model.run([2, 5]).reshape(-1).tolist(), [26])
        model.close()
        self.assertIn(resident_id, dev.protocol.transport.buffers)
        pool.close()
        self.assertNotIn(resident_id, dev.protocol.transport.buffers)

    def test_model_run_writes_input_executes_graph_and_reads_output(self):
        dev = RecordingDevice()
        dev.freed = []
        model = Model(dev, descriptor_tensor="desc", input_tensor="input", output_tensor="output")
        result = model.run([9])

        self.assertEqual(len(dev.writes), 1)
        self.assertEqual(dev.writes[0][0], "input")
        self.assertEqual(dev.graph_calls, [("desc", "input", "output", 0)])
        self.assertEqual(result, dev.output)

    def test_model_close_releases_descriptor_and_model_tensors_once(self):
        dev = RecordingDevice()
        dev.freed = []
        model = Model(dev, descriptor_tensor="desc", input_tensor="input", output_tensor="output", tensors=["input", "output"])
        model.close()
        model.close()
        self.assertEqual(dev.freed, ["desc", "output", "input"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.run([1])

    def test_model_run_executes_mock_graph_relu(self):
        dev = Device.mock()
        package = ModelPackage(
            buffers=[
                HostBuffer("input", (4,), DType.INT8, usage=BufferUsage.ACTIVATION),
                HostBuffer("output", (4,), DType.INT8, usage=BufferUsage.OUTPUT),
            ],
            layers=[Layer(OpType.ACTIVATION, input0=0, output=1)],
            input_index=0,
            output_index=1,
        )
        model = upload_model(dev, package)
        result = model.run([-3, 0, 5, -128])
        values = result.tolist() if hasattr(result, "tolist") else result
        self.assertEqual(values, [0, 0, 5, 0])

    def test_model_run_executes_mock_upsample_then_concat(self):
        dev = Device.mock()
        package = ModelPackage(
            buffers=[
                HostBuffer("input", (1, 1, 2, 2), DType.INT8, usage=BufferUsage.ACTIVATION),
                HostBuffer.from_values("skip", (1, 1, 4, 4), DType.INT8, list(range(10, 26)), usage=BufferUsage.ACTIVATION),
                HostBuffer("upsampled", (1, 1, 4, 4), DType.INT8, usage=BufferUsage.ACTIVATION),
                HostBuffer("output", (1, 2, 4, 4), DType.INT8, usage=BufferUsage.OUTPUT),
            ],
            layers=[
                Layer(OpType.UPSAMPLE, input0=0, output=2, n=1, c=1, h=2, w=2, out_h=4, out_w=4),
                Layer(OpType.CONCAT, input0=2, input1=1, output=3, n=1, h=4, w=4, out_c=2),
            ],
            input_index=0,
            output_index=3,
        )
        model = upload_model(dev, package)
        result = model.run([1, 2, 3, 4])
        values = result.reshape(-1).tolist() if hasattr(result, "reshape") else result
        self.assertEqual(values[:16], [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4])
        self.assertEqual(values[16:], list(range(10, 26)))

    def test_model_run_executes_tiny_conv_relu_package(self):
        dev = Device.mock()
        package = build_tiny_conv_relu_package((1, 1, 3, 3), [[[[1, 0], [0, 1]]]], bias_values=[-10])
        model = upload_model(dev, package)
        result = model.run([1, 2, 3, 4, 5, 6, 7, 8, 9])
        values = result.reshape(-1).tolist() if hasattr(result, "reshape") else result
        self.assertEqual(values, [0, 0, 2, 4])

    def test_model_run_executes_mock_requantize(self):
        dev = Device.mock()
        package = ModelPackage(
            buffers=[
                HostBuffer.from_values("input", (5,), DType.INT32, [-300, -2, 0, 8, 500], usage=BufferUsage.ACTIVATION),
                HostBuffer.from_values("scale", (4,), DType.INT32, [1, 1, 0, 3], usage=BufferUsage.WEIGHT),
                HostBuffer("output", (5,), DType.INT8, usage=BufferUsage.OUTPUT),
            ],
            layers=[Layer(OpType.REQUANTIZE, input0=0, output=2, scale=1)],
            input_index=0,
            output_index=2,
        )
        model = upload_model(dev, package)
        result = model.run([-300, -2, 0, 8, 500])
        values = result.tolist() if hasattr(result, "tolist") else result
        self.assertEqual(values, [-128, 2, 3, 7, 127])

    def test_model_run_executes_mock_power_of_two_leaky_relu(self):
        dev = Device.mock()
        package = ModelPackage(
            buffers=[HostBuffer("input", (4,), DType.INT8), HostBuffer("output", (4,), DType.INT8, usage=BufferUsage.OUTPUT)],
            layers=[Layer(OpType.ACTIVATION, input0=0, output=1, flags=1 | (2 << 8))],
            input_index=0,
            output_index=1,
        )
        model = upload_model(dev, package)
        result = model.run([-8, -3, 0, 7])
        values = result.tolist() if hasattr(result, "tolist") else result
        self.assertEqual(values, [-2, -1, 0, 7])


if __name__ == "__main__":
    unittest.main()
