"""Minimal offline compatibility surface needed by PYNQ's overlay state code.

PYNQ 3.0 imports :class:`pydantic.BaseModel` solely for two small metadata
records.  Some legacy PYNQ-Z2 images omit the pydantic package and cannot
reach PyPI.  This module is intentionally limited to the constructor and
JSON helpers used by ``pynq.pl_server.global_state``; it is not a replacement
for pydantic in application code.
"""

from __future__ import annotations

import json


class BaseModel:
    """Small subset of pydantic.BaseModel required by PYNQ global state."""

    def __init__(self, **kwargs):
        for name in self.__annotations__:
            if name in kwargs:
                value = kwargs[name]
            else:
                value = getattr(type(self), name, None)
            setattr(self, name, value)

    @classmethod
    def parse_obj(cls, value):
        return cls(**value)

    def dict(self):
        def encode(value):
            if isinstance(value, BaseModel):
                return value.dict()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return value

        return {name: encode(getattr(self, name)) for name in self.__annotations__}

    def json(self):
        return json.dumps(self.dict())
