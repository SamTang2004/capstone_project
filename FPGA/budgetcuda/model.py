from __future__ import annotations

import time


class Model:
    def __init__(self, device, descriptor_tensor, input_tensor, output_tensor, weights=None, tensors=None, package=None):
        self.device = device
        self.descriptor_tensor = descriptor_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.weights = list(weights or [])
        self.tensors = list(tensors or [])
        self.package = package
        self._closed = False

    def run(self, input_np, diagnostics=None):
        if self._closed:
            raise RuntimeError("Model has been closed")
        if diagnostics is not None:
            return self._run_with_diagnostics(input_np, diagnostics)
        self.device.write(self.input_tensor, input_np)
        self.device.execute_graph(self.descriptor_tensor, self.input_tensor, self.output_tensor)
        return self.device.read(self.output_tensor)

    def _run_with_diagnostics(self, input_np, settings):
        from .diagnostics import DiagnosticResult, DiagnosticsSettings, parse_diagnostics

        if not isinstance(settings, DiagnosticsSettings):
            raise TypeError("diagnostics must be a DiagnosticsSettings instance")
        if self.package is None:
            raise RuntimeError("Diagnostics require a model loaded from a ModelPackage")

        report_bytes = settings.buffer_bytes(len(self.package.layers))
        report_tensor = self.device.alloc((report_bytes,), "int8", usage="workspace")
        try:
            self.device.write(report_tensor, settings.initial_buffer(len(self.package.layers)))
            host_started = time.perf_counter_ns()

            started = time.perf_counter_ns()
            self.device.write(self.input_tensor, input_np)
            host_write_ns = time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            self.device.execute_graph(
                self.descriptor_tensor,
                self.input_tensor,
                self.output_tensor,
                flags=settings.flags,
                diagnostics_tensor=report_tensor,
            )
            host_execute_ns = time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            output = self.device.read(self.output_tensor)
            host_read_ns = time.perf_counter_ns() - started
            host_total_ns = time.perf_counter_ns() - host_started

            raw_report = self.device.read(report_tensor)
            report = parse_diagnostics(raw_report, settings)
            report.host_write_ns = host_write_ns
            report.host_execute_ns = host_execute_ns
            report.host_read_ns = host_read_ns
            report.host_total_ns = host_total_ns
            return DiagnosticResult(output, report)
        finally:
            self.device.free(report_tensor)

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
