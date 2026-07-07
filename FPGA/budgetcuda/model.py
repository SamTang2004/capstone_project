from __future__ import annotations


class Model:
    def __init__(self, device, descriptor_tensor, input_tensor, output_tensor, weights=None):
        self.device = device
        self.descriptor_tensor = descriptor_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.weights = list(weights or [])

    def run(self, input_np):
        self.device.write(self.input_tensor, input_np)
        self.device.execute_graph(self.descriptor_tensor, self.input_tensor, self.output_tensor)
        return self.device.read(self.output_tensor)
