# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ctypes import c_void_p

import torch

from vllm.device_allocator.storage_tiers.base import BaseStorageTier
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.utils.platform_utils import is_pin_memory_available

libcudart = CudaRTLibrary()


class RamStorageTier(BaseStorageTier):
    def __init__(self) -> None:
        self.cpu_backup_tensors: dict[int, torch.Tensor] = {}

    def offload(self, cuda_ptr: int, size_in_bytes: int) -> None:
        cpu_backup_tensor = torch.empty(
            size_in_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        libcudart.cudaMemcpy(
            c_void_p(cpu_backup_tensor.data_ptr()),
            c_void_p(cuda_ptr),
            size_in_bytes,
        )
        self.cpu_backup_tensors[cuda_ptr] = cpu_backup_tensor

    def load(self, cuda_ptr: int) -> None:
        cpu_backup_tensor = self.cpu_backup_tensors.pop(cuda_ptr, None)
        if cpu_backup_tensor is None:
            return
        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
        libcudart.cudaMemcpy(
            c_void_p(cuda_ptr),
            c_void_p(cpu_backup_tensor.data_ptr()),
            size_in_bytes,
        )

    def free(self, cuda_ptr: int) -> None:
        _ = self.cpu_backup_tensors.pop(cuda_ptr, None)
