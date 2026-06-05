# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVMe offload tier for vLLM sleep mode.

Pipelined wake: ``N`` ``pread`` reader threads feed a copier that does
non-blocking ``cudaMemcpyAsync`` on ``N`` round-robin CUDA streams.
Sleep commits the on-disk weight file atomically (``.tmp -> .bin`` +
``.valid`` marker) and is the write-once cache: cycle~1 pays the
NVMe write cost, cycles 2+ skip the write and replay layout offsets
only.

Cache key: each ``begin_sleep`` requires a model identifier so the
on-disk file is uniquely addressable across models. The identifier
comes from the API ``model_id`` argument (preferred) or the
``VLLM_NVME_CACHE_KEY`` env var (fallback for callers that have not
been plumbed through the API). Each file also embeds a build
fingerprint so a fork-commit bump invalidates stale caches without
manual cleanup.

Env vars:
    VLLM_NVME_OFFLOAD_DIR    cache directory (default /tmp)
    VLLM_NVME_CACHE_KEY      cache-key fallback when the API caller
                             does not pass ``model_id``
    VLLM_NVME_FINGERPRINT    fingerprint override; defaults to
                             ``vllm.__version__``
"""
import ctypes
import logging
import os
import queue
import threading
from contextlib import suppress
from ctypes import c_size_t, c_uint, c_void_p
from typing import Any

import torch

from vllm.device_allocator.storage_tiers.base import BaseStorageTier
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

logger = logging.getLogger(__name__)
libcudart = CudaRTLibrary()
_BLOCK = 4096                       # O_DIRECT alignment.
_H2D = c_uint(4)                    # cudaMemcpyDefault.
_NONBLOCK = c_uint(1)               # cudaStreamNonBlocking.
_NUM_BUFS_MULT = 2                  # buffers per reader (one full, one free).
_CHUNK_BYTES = 32 * 1024 * 1024     # 32 MiB; sits in cuMem cheap regime
                                    # (>2 MiB granularity floor, <1 GiB cliff).
_NUM_READERS = 4                    # reader thread-pool size for the
                                    # wake pipeline (overlaps pread with
                                    # the per-stream H2D copies).
_HUGEPAGE_SIZE = 2 * 1024 * 1024
_MAP_HUGETLB = 0x40000
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_PROT_RW = 0x03
_MAP_FAILED = ctypes.c_void_p(-1).value & ((1 << 64) - 1)


def _align_up(n: int, a: int = _BLOCK) -> int:
    return (n + a - 1) // a * a


def _safe_model_id(m: str) -> str:
    return m.replace("/", "_").replace(os.sep, "_")


# CudaRTLibrary exposes the sync cudaMemcpy; resolve the async/stream
# entry points lazily against the same libcudart.
_cuda: dict[str, Any] | None = None
_libc: ctypes.CDLL | None = None


def _get_cuda_async() -> dict[str, Any]:
    global _cuda
    if _cuda is not None:
        return _cuda
    sig = {
        "memcpy_async": ("cudaMemcpyAsync",
                         [c_void_p, c_void_p, c_size_t, c_uint, c_void_p]),
        "stream_create": ("cudaStreamCreateWithFlags",
                          [ctypes.POINTER(c_void_p), c_uint]),
        "stream_sync":   ("cudaStreamSynchronize", [c_void_p]),
        "stream_destroy": ("cudaStreamDestroy", [c_void_p]),
        "host_register": ("cudaHostRegister",
                          [c_void_p, c_size_t, c_uint]),
        "host_unregister": ("cudaHostUnregister", [c_void_p]),
    }
    fns: dict[str, Any] = {}
    for k, (name, argtypes) in sig.items():
        fn = getattr(libcudart.lib, name)
        fn.restype, fn.argtypes = c_uint, argtypes
        fns[k] = fn
    _cuda = fns
    return fns


def _get_libc() -> ctypes.CDLL:
    global _libc
    if _libc is not None:
        return _libc
    L = ctypes.CDLL("libc.so.6", use_errno=True)
    L.mmap.restype = c_void_p
    L.mmap.argtypes = [c_void_p, c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int64]
    L.munmap.restype = ctypes.c_int
    L.munmap.argtypes = [c_void_p, c_size_t]
    _libc = L
    return L


class _HugepageBuf:
    """2 MiB-page-backed pinned buffer; mimics torch tensor's
    ``data_ptr()`` / ``numpy()`` API for the wake pipeline.

    The mapping is page-locked via ``cudaHostRegister`` so it can serve
    as a copy source/dest like a torch ``pin_memory`` tensor.
    """
    __slots__ = ("_addr", "_mapped", "_registered", "_view")

    def __init__(self, size: int):
        self._addr, self._registered = 0, False
        mapped = ((size + _HUGEPAGE_SIZE - 1)
                  // _HUGEPAGE_SIZE) * _HUGEPAGE_SIZE
        libc = _get_libc()
        addr = libc.mmap(None, mapped, _PROT_RW,
                         _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB,
                         -1, 0)
        if addr is None or (addr & ((1 << 64) - 1)) == _MAP_FAILED:
            e = ctypes.get_errno()
            raise OSError(e, f"mmap(MAP_HUGETLB,{mapped}): "
                          f"{os.strerror(e)}")
        self._addr, self._mapped = addr, mapped
        ret = _get_cuda_async()["host_register"](
            c_void_p(addr), c_size_t(mapped), c_uint(0))
        if ret != 0:
            libc.munmap(c_void_p(addr), c_size_t(mapped))
            self._addr = 0
            raise RuntimeError(f"cudaHostRegister failed: cudaError={ret}")
        self._registered = True
        # numpy view over the same bytes so the existing preadv() call
        # site does not need to special-case hugepage bufs.
        self._view = (ctypes.c_uint8 * size).from_address(addr)

    def data_ptr(self) -> int:
        return self._addr

    def numpy(self):
        import numpy as np
        return np.frombuffer(self._view, dtype=np.uint8)

    def free(self) -> None:
        if self._addr == 0:
            return
        with suppress(Exception):
            if self._registered:
                _get_cuda_async()["host_unregister"](c_void_p(self._addr))
        with suppress(Exception):
            _get_libc().munmap(c_void_p(self._addr), c_size_t(self._mapped))
        self._addr = 0

    def __del__(self):
        with suppress(Exception):
            self.free()


def _alloc_hugepage_pool(chunk: int, count: int) -> list | None:
    """Try *count* hugepage buffers; warn and return None on failure
    so the caller can fall back to torch pin_memory."""
    bufs: list[_HugepageBuf] = []
    try:
        for _ in range(count):
            bufs.append(_HugepageBuf(chunk))
        return bufs
    except (OSError, RuntimeError) as e:
        for b in bufs:
            b.free()
        logger.warning("NvmeStorageTier: hugepage alloc failed (%d x %d "
                       "MiB): %s -- falling back to torch pin_memory",
                       count, chunk // (1024 * 1024), e)
        return None


class NvmeStorageTier(BaseStorageTier):
    def __init__(self) -> None:
        self._offload_dir = os.environ.get("VLLM_NVME_OFFLOAD_DIR", "/tmp")
        self._chunk = _CHUNK_BYTES
        self._n = _NUM_READERS

        # Bounce-buffer pool. Prefer MAP_HUGETLB-backed buffers
        # (2 MiB pages, cudaHostRegister-pinned) so the wake-pipeline
        # preadv() reads land on fewer TLB-distinct pages — measurable
        # gain at 32 MiB chunks. Falls back to cudaHostAlloc-backed
        # torch pin_memory (also page-aligned, O_DIRECT-safe) when no
        # hugepages are reserved.
        n_bufs = max(2, _NUM_BUFS_MULT * self._n)
        hp = _alloc_hugepage_pool(self._chunk, n_bufs)
        if hp is not None:
            self._pool: list[Any] = hp
        else:
            self._pool = [
                torch.empty(self._chunk, dtype=torch.uint8,
                            device="cpu", pin_memory=True)
                for _ in range(n_bufs)
            ]
        self._wbuf_ptr = self._pool[0].data_ptr()
        self._wbuf_view = self._pool[0].numpy()

        # File state — populated by begin_sleep before any I/O.
        self._fd = -1
        self._final_path: str = ""
        self._tmp_path: str = ""
        self._valid_path: str = ""
        self._cache_hit = False
        self._handles: dict[int, tuple[int, int]] = {}  # ptr -> (off, sz)
        self._next_off = 0

        # Wake-pipeline state.
        self._work_q: queue.Queue | None = None
        self._full_q: queue.Queue | None = None
        self._free_q: queue.Queue | None = None
        self._readers: list[threading.Thread] = []
        self._copier: threading.Thread | None = None
        self._done: threading.Event | None = None
        self._device = -1
        self._err: list[str | None] = [None, None]   # [io, copy]

    def _cache_present(self) -> bool:
        return (os.path.exists(self._final_path)
                and os.path.exists(self._valid_path))

    @staticmethod
    def _resolve_cache_key(model_id: str | None) -> str:
        key = model_id or os.environ.get("VLLM_NVME_CACHE_KEY")
        if not key:
            raise ValueError(
                "NvmeStorageTier.begin_sleep requires a model identifier: "
                "pass `model_id` via the API or set VLLM_NVME_CACHE_KEY env "
                "var. The anonymous (O_TMPFILE, write-every-cycle) path is "
                "no longer supported.")
        return key

    @staticmethod
    def _resolve_fingerprint(fingerprint: str | None) -> str:
        if fingerprint:
            return fingerprint
        override = os.environ.get("VLLM_NVME_FINGERPRINT")
        if override:
            return override
        # Last resort: vllm package version. Stable across runs of the
        # same build; bumped automatically when the fork moves to a new
        # vLLM version, so stale caches are invalidated without manual
        # cleanup.
        from vllm import __version__ as _vllm_version
        return _vllm_version

    # -- sleep ---------------------------------------------------------
    def begin_sleep(self, model_id: str | None = None,
                    fingerprint: str | None = None) -> None:
        # Reset layout so cache-replay offsets match the disk format.
        self._handles.clear()
        self._next_off = 0
        self._cache_hit = False
        key = self._resolve_cache_key(model_id)
        fp = self._resolve_fingerprint(fingerprint)
        stem = os.path.join(
            self._offload_dir,
            f"{_safe_model_id(key)}.{fp}.weights.bin")
        self._final_path = stem
        self._tmp_path = stem + ".tmp"
        self._valid_path = stem + ".valid"
        if self._cache_present():
            # Cache hit: offload() records layout only; no I/O.
            self._cache_hit = True
            return
        # Cache miss: scrub orphans so begin_wake's .bin + .valid gate
        # stays meaningful after a crashed prior run.
        for stale in (self._tmp_path, self._valid_path):
            with suppress(FileNotFoundError):
                os.unlink(stale)
        self._fd = os.open(self._tmp_path,
                           os.O_CREAT | os.O_RDWR | os.O_DIRECT | os.O_TRUNC,
                           0o600)

    def offload(self, cuda_ptr: int, size_in_bytes: int) -> None:
        # Sync D2H into a pinned buffer (cumem unmaps the source right
        # after offload returns, so the GPU read must be done by then),
        # then sync pwrite to file.
        start = self._next_off
        gpu, off, rem = cuda_ptr, start, size_in_bytes
        while rem > 0:
            ch = min(rem, self._chunk)
            al = _align_up(ch)
            if not self._cache_hit:
                libcudart.cudaMemcpy(c_void_p(self._wbuf_ptr),
                                     c_void_p(gpu), ch)
                if al > ch:  # Zero O_DIRECT tail for deterministic file.
                    ctypes.memset(self._wbuf_ptr + ch, 0, al - ch)
                n = os.pwrite(self._fd, self._wbuf_view[:al], off)
                if n != al:
                    raise OSError(f"short pwrite: {n}/{al} at {off}")
            gpu, off, rem = gpu + ch, off + al, rem - ch
        self._next_off = off
        self._handles[cuda_ptr] = (start, size_in_bytes)

    def end_sleep(self) -> None:
        if self._cache_hit:
            return
        # Atomic commit: fsync -> rename .tmp -> .bin -> write+fsync .valid.
        # A crash between rename and the .valid write is recovered by the
        # next begin_sleep's orphan scrub.
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = -1
        os.rename(self._tmp_path, self._final_path)
        vfd = os.open(self._valid_path,
                      os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.write(vfd, b"OK\n")
            os.fsync(vfd)
        finally:
            os.close(vfd)

    # -- wake ----------------------------------------------------------
    def begin_wake(self) -> None:
        if not self._cache_present():
            raise RuntimeError(
                f"NvmeStorageTier.begin_wake: cached file missing "
                f"({self._final_path} or .valid); sleep interrupted?")
        if self._fd < 0:
            self._fd = os.open(self._final_path,
                               os.O_RDONLY | os.O_DIRECT)
        self._free_q = queue.Queue()
        for buf in self._pool:
            self._free_q.put(buf)
        self._full_q = queue.Queue()
        self._work_q = queue.Queue()
        self._err = [None, None]
        self._done = threading.Event()
        self._device = torch.cuda.current_device()
        self._readers = [threading.Thread(target=self._reader_fn, daemon=True)
                         for _ in range(self._n)]
        for t in self._readers:
            t.start()
        self._copier = threading.Thread(target=self._copier_fn, daemon=True)
        self._copier.start()

    def load(self, cuda_ptr: int) -> None:
        h = self._handles.get(cuda_ptr)
        if h is None:
            return
        assert self._work_q is not None
        gpu, off, rem = cuda_ptr, h[0], h[1]
        while rem > 0:
            ch = min(rem, self._chunk)
            self._work_q.put((gpu, off, ch))
            gpu, off, rem = gpu + ch, off + _align_up(ch), rem - ch

    def end_wake(self) -> None:
        assert self._work_q is not None and self._done is not None
        for _ in range(self._n):
            self._work_q.put(None)
        if not self._done.wait(timeout=300):
            raise RuntimeError("NvmeStorageTier.end_wake: copier timeout")
        for t in self._readers:
            t.join(timeout=30)
        if self._copier is not None:
            self._copier.join(timeout=30)
        if self._err[0]:
            raise RuntimeError(f"NVMe wake I/O: {self._err[0]}")
        if self._err[1]:
            raise RuntimeError(f"NVMe wake copy: {self._err[1]}")
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _reader_fn(self) -> None:
        assert self._work_q is not None and self._free_q is not None
        assert self._full_q is not None
        try:
            while True:
                item = self._work_q.get()
                if item is None:
                    self._full_q.put(None)
                    return
                gpu_dst, file_off, nbytes = item
                al = _align_up(nbytes)
                buf = self._free_q.get()
                view = buf.numpy()[:al]
                n = os.preadv(self._fd, [view], file_off)
                if n != al:
                    self._err[0] = f"short preadv {n}/{al} at {file_off}"
                    self._full_q.put(None)
                    return
                self._full_q.put((buf, gpu_dst, nbytes))
        except Exception as e:
            self._err[0] = str(e)
            self._full_q.put(None)

    def _copier_fn(self) -> None:
        """Drain full_q with cudaMemcpyAsync on N round-robin
        non-blocking CUDA streams; reclaim each stream's prior buffer
        before reusing it so in-flight H2D is bounded by N streams."""
        assert self._full_q is not None and self._free_q is not None
        assert self._done is not None
        cuda = _get_cuda_async()
        streams: list[c_void_p] = []
        for _ in range(self._n):
            s = c_void_p()
            cuda["stream_create"](ctypes.byref(s), _NONBLOCK)
            streams.append(s)
        in_flight: list[Any] = [None] * self._n
        try:
            torch.cuda.set_device(self._device)
            done = 0
            i = 0
            while done < self._n:
                item = self._full_q.get()
                if item is None:
                    done += 1
                    continue
                buf, gpu_dst, nbytes = item
                # Reclaim prior buf on this stream before reuse.
                if in_flight[i] is not None:
                    cuda["stream_sync"](streams[i])
                    self._free_q.put(in_flight[i])
                cuda["memcpy_async"](
                    c_void_p(gpu_dst), c_void_p(buf.data_ptr()),
                    c_size_t(nbytes), _H2D, streams[i])
                in_flight[i] = buf
                i = (i + 1) % self._n
            for k, buf in enumerate(in_flight):
                if buf is not None:
                    cuda["stream_sync"](streams[k])
                    self._free_q.put(buf)
        except Exception as e:
            self._err[1] = str(e)
        finally:
            for s in streams:
                with suppress(Exception):
                    cuda["stream_destroy"](s)
            self._done.set()

    def free(self, cuda_ptr: int) -> None:
        self._handles.pop(cuda_ptr, None)

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with suppress(Exception):
                os.close(fd)

    __del__ = close
