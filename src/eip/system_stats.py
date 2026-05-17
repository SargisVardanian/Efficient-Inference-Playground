from __future__ import annotations

import os
from dataclasses import dataclass

import psutil


@dataclass
class MemorySnapshot:
    rss_mb: float
    system_used_mb: float
    system_available_mb: float


def current_memory() -> MemorySnapshot:
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    return MemorySnapshot(
        rss_mb=process.memory_info().rss / (1024 * 1024),
        system_used_mb=vm.used / (1024 * 1024),
        system_available_mb=vm.available / (1024 * 1024),
    )

