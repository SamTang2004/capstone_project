from __future__ import annotations


class Model:
    def __init__(self, device, descriptor_tensor, input_tensor, output_tensor, weights=None, tensors=None):
        self.device = device
        self.descriptor_tensor = descriptor_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.weights = list(weights or [])
        self.tensors = list(tensors or [])
        self._closed = False

    def run(self, input_np):
        if self._closed:
            raise RuntimeError("Model has been closed")
        self.device.write(self.input_tensor, input_np)
        self.device.execute_graph(self.descriptor_tensor, self.input_tensor, self.output_tensor)
        return self.device.read(self.output_tensor)

    def close(self):
        """Release the descriptor and all model buffers from board DRAM."""
        if self._closed:
            return
        self._closed = True
        first_error = None
        for tensor in [self.descriptor_tensor, *reversed(self.tensors)]:
            try:
                self.device.free(tensor)
            except Exception as exc:  # Preserve cleanup of every remaining buffer.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
