from __future__ import annotations

from time import perf_counter_ns

from .diagnostics import DiagnosticResult, DiagnosticsSettings, RunDiagnostics, parse_diagnostics

class Model:
    def __init__(self, device, descriptor_tensor, input_tensor, output_tensor, weights=None, tensors=None, package=None):
        self.device = device
        self.descriptor_tensor = descriptor_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.weights = list(weights or [])
        self.tensors = list(tensors or [])
        self.package = package
        self._diagnostics_tensor = None
        self._closed = False

    def run(self, input_np, diagnostics=None):
        if self._closed:
            raise RuntimeError("Model has been closed")
        settings = DiagnosticsSettings() if diagnostics is True else diagnostics
        if settings is not None and not isinstance(settings, DiagnosticsSettings):
            raise TypeError("diagnostics must be True or DiagnosticsSettings")
        started = perf_counter_ns()
        write_started = perf_counter_ns()
        self.device.write(self.input_tensor, input_np)
        write_ns = perf_counter_ns() - write_started
        if settings is None:
            self.device.execute_graph(self.descriptor_tensor, self.input_tensor, self.output_tensor)
            return self.device.read(self.output_tensor)
        if self.package is None:
            raise ValueError("Diagnostics require the uploaded model package")
        initial = settings.initial_buffer(len(self.package.layers))
        if self._diagnostics_tensor is None or self._diagnostics_tensor.nbytes != len(initial):
            if self._diagnostics_tensor is not None:
                self.device.free(self._diagnostics_tensor)
            self._diagnostics_tensor = self.device.alloc(
                (len(initial),), "int8", usage="workspace")
        self.device.write(self._diagnostics_tensor, initial)
        execute_started = perf_counter_ns()
        self.device.execute_graph(
            self.descriptor_tensor,
            self.input_tensor,
            self.output_tensor,
            settings.flags,
            self._diagnostics_tensor,
        )
        execute_ns = perf_counter_ns() - execute_started
        read_started = perf_counter_ns()
        output = self.device.read(self.output_tensor)
        report = parse_diagnostics(
            self.device.read(self._diagnostics_tensor), settings)
        read_ns = perf_counter_ns() - read_started
        report = RunDiagnostics(settings, report.layers, report.firmware_total_ns, report.arena_bytes_used, write_ns, execute_ns, read_ns, perf_counter_ns() - started)
        return DiagnosticResult(output, report)

    def close(self):
        """Release the descriptor and all model buffers from board DRAM."""
        if self._closed:
            return
        self._closed = True
        first_error = None
        for tensor in [self._diagnostics_tensor, self.descriptor_tensor, *reversed(self.tensors)]:
            if tensor is None:
                continue
            try:
                self.device.free(tensor)
            except Exception as exc:  # Preserve cleanup of every remaining buffer.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
