"""
instrument.py
─────────────
Drop-in instrumentation wrapper for energy, latency, and GPU profiling.
Works with Zeus (GPU) and PyRAPL (CPU/DRAM). Gracefully degrades if
either library is not installed — falls back to wall-clock only.

Usage (wrap any callable):
    from instrument import ResourceMonitor

    monitor = ResourceMonitor()
    with monitor.measure() as record:
        result = your_pipeline_call(...)
    record.finalize()
    print(record.to_dict())
"""

import time
import threading
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    import pyRAPL
    pyRAPL.setup()
    _PYRAPL_AVAILABLE = True
except Exception:
    _PYRAPL_AVAILABLE = False
    print("[instrument] PyRAPL not available — CPU energy will be None.")

try:
    from zeus.monitor import ZeusMonitor
    _ZEUS_AVAILABLE = True
except Exception:
    _ZEUS_AVAILABLE = False
    print("[instrument] Zeus not available — GPU energy will be None.")

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


# ── Data class for a single measurement record ───────────────────────────────
@dataclass
class ResourceRecord:
    config_id:         str  = ""
    query_id:          int  = 0
    schema_id:         str  = ""
    retrieval_mode:    str  = ""

    # Timing
    latency_ms:        float | None = None

    # GPU (Zeus)
    gpu_energy_mj:     float | None = None   # millijoules
    gpu_util_pct:      float | None = None   # mean % during call
    gpu_mem_mb:        float | None = None   # peak MB

    # CPU/DRAM (PyRAPL)
    cpu_energy_mj:     float | None = None
    dram_energy_mj:    float | None = None

    # Tokens
    prompt_tokens:     int   | None = None
    completion_tokens: int   | None = None

    # Reliability
    repair_count:      int   = 0
    json_valid:        bool  = False
    schema_valid:      bool  = False
    alignment_score:   float | None = None
    structural_score:  float | None = None

    # Raw outputs for debugging
    error:             str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── GPU utilisation poller (runs in background thread) ───────────────────────
class _GpuPoller:
    """Polls NVML at 100ms intervals to get mean utilisation and peak memory."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._running      = False
        self._thread       = None
        self.util_samples  : list[float] = []
        self.mem_samples   : list[float] = []

    def start(self):
        if not _NVML_AVAILABLE:
            return
        self._running = True
        self.util_samples = []
        self.mem_samples  = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        while self._running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.util_samples.append(float(util.gpu))
                self.mem_samples.append(float(mem.used) / 1024 / 1024)  # bytes → MB
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self) -> tuple[float | None, float | None]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.util_samples:
            mean_util = sum(self.util_samples) / len(self.util_samples)
            peak_mem  = max(self.mem_samples) if self.mem_samples else None
            return mean_util, peak_mem
        return None, None


# ── Main monitor class ───────────────────────────────────────────────────────
class ResourceMonitor:
    """
    Usage:
        monitor = ResourceMonitor(gpu_index=0)

        record = ResourceRecord(config_id="C2", query_id=42, ...)
        with monitor.measure(record):
            output = pipeline.run(query)
        # record is now populated with energy/latency data
    """

    def __init__(self, gpu_index: int = 0):
        self._gpu_index = gpu_index
        self._zeus      = None
        if _ZEUS_AVAILABLE:
            try:
                self._zeus = ZeusMonitor(gpu_indices=[gpu_index])
            except Exception as e:
                print(f"[instrument] Zeus init failed: {e}")

    @contextlib.contextmanager
    def measure(self, record: ResourceRecord):
        """Context manager. Populates record in-place."""
        poller = _GpuPoller(self._device_index if hasattr(self, '_device_index')
                             else self._gpu_index)

        # ── PyRAPL measurement ───────────────────────────────────────────────
        rapl_meter = None
        if _PYRAPL_AVAILABLE:
            rapl_meter = pyRAPL.Measurement("query")
            rapl_meter.begin()

        # ── Zeus measurement ─────────────────────────────────────────────────
        zeus_label = f"query_{record.query_id}_{record.config_id}"
        if self._zeus:
            try:
                self._zeus.begin_window(zeus_label)
            except Exception:
                pass

        # ── NVML poller ──────────────────────────────────────────────────────
        poller.start()

        t_start = time.perf_counter()

        try:
            yield record          # ← user code runs here
        finally:
            t_end = time.perf_counter()
            record.latency_ms = (t_end - t_start) * 1000.0

            # Stop poller
            mean_util, peak_mem = poller.stop()
            record.gpu_util_pct = mean_util
            record.gpu_mem_mb   = peak_mem

            # Zeus
            if self._zeus:
                try:
                    measurement = self._zeus.end_window(zeus_label)
                    # Zeus returns total_energy in Joules per GPU
                    joules = measurement.total_energy.get(self._gpu_index, None)
                    if joules is not None:
                        record.gpu_energy_mj = joules * 1000.0  # J → mJ
                except Exception as e:
                    record.error += f"[zeus] {e}; "

            # PyRAPL
            if rapl_meter:
                try:
                    rapl_meter.end()
                    pkg = rapl_meter.result.pkg
                    dram = rapl_meter.result.dram
                    # PyRAPL returns µJ → convert to mJ
                    if pkg:
                        record.cpu_energy_mj  = sum(pkg)  / 1000.0
                    if dram:
                        record.dram_energy_mj = sum(dram) / 1000.0
                except Exception as e:
                    record.error += f"[pyrapl] {e}; "


# ── Convenience: estimate token count without tiktoken ───────────────────────
def estimate_tokens(text: str) -> int:
    """Rough approximation: 1 token ≈ 4 chars (good enough for logging)."""
    return max(1, len(text) // 4)
