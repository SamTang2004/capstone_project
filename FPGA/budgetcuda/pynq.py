from __future__ import annotations

from .compiler import ModelPackage, upload_model
from .device import Device


class PynqDevice(Device):
    def __init__(self, address: str, port: int = 9000):
        super().__init__(address=address, port=port)

    def load_model(self, package: ModelPackage):
        return upload_model(self, package)


def connect(address: str, port: int = 9000) -> PynqDevice:
    return PynqDevice(address, port)
