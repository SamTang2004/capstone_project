import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device
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


class ModelTest(unittest.TestCase):
    def test_model_run_writes_input_executes_graph_and_reads_output(self):
        dev = RecordingDevice()
        model = Model(dev, descriptor_tensor="desc", input_tensor="input", output_tensor="output")
        result = model.run([9])

        self.assertEqual(len(dev.writes), 1)
        self.assertEqual(dev.writes[0][0], "input")
        self.assertEqual(dev.graph_calls, [("desc", "input", "output", 0)])
        self.assertEqual(result, dev.output)


if __name__ == "__main__":
    unittest.main()
