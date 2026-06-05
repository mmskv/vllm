# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Literal

StorageTierName = Literal["ram", "nvme"]


class BaseStorageTier:
    """Base class for sleep-mode storage tiers.

    Lifecycle of a sleep/wake cycle:

        tier = make_storage_tier(name)
        tier.begin_sleep(model_id, fingerprint)  # open backing file
        for ptr, size in tensors:
            tier.offload(ptr, size)              # may be skipped on cache hit
        tier.end_sleep()                         # commit (atomic for named)
        tier.begin_wake()                        # spawn any I/O pipeline
        for ptr in tensors:
            tier.load(ptr)                       # may be async
        tier.end_wake()                          # sync barrier
        tier.close()                             # release resources

    `load()` may be asynchronous: tiers are free to enqueue work and
    return immediately. `end_wake()` is the sync barrier; after it
    returns, the GPU tensors are guaranteed to be valid.

    `begin_sleep`/`end_sleep` are the sleep-side equivalents. Tiers that
    do not need persistence (e.g. RAM) may treat them as no-ops; the
    NVMe tier uses them to switch between anonymous (O_TMPFILE) and
    named (write-once cache, atomic commit) modes.
    """

    def offload(self, _cuda_ptr: int, _size_in_bytes: int) -> None: ...
    def load(self, _cuda_ptr: int) -> None: ...
    def free(self, _cuda_ptr: int) -> None: ...

    def begin_sleep(
        self,
        model_id: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        """Prepare the tier to receive offload() calls."""

    def end_sleep(self) -> None:
        """Commit / finalize anything written during this sleep cycle."""

    def begin_wake(self) -> None:
        """Open any resources needed for loading."""

    def end_wake(self) -> None:
        """Sync barrier: caller can use GPU tensors after this returns."""

    def close(self) -> None:
        """Release any process-lifetime resources."""


_TIER_REGISTRY: dict[StorageTierName, tuple[str, str]] = {
    "ram": ("vllm.device_allocator.storage_tiers.ram", "RamStorageTier"),
    "nvme": ("vllm.device_allocator.storage_tiers.nvme", "NvmeStorageTier"),
}


def make_storage_tier(name: StorageTierName) -> BaseStorageTier:
    module_path, class_name = _TIER_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
