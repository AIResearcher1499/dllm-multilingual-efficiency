"""Hardware provenance, recorded on every row.

Wall-clock numbers from different machines must never be compared, and once a
jsonl file has been merged there is no way to enforce that unless each row says
where it came from. Every quantity in this project is either hardware-invariant
(NFE, accuracy, distinctness, parallel_factor) or hardware-dependent
(wall_clock, tokens per second). Mixing the second kind silently is the easiest
way to publish a wrong number.

`concurrent_procs` exists for the two-GPU box: running two cells at once is free
for NFE and quality, and contaminates wall-clock through CPU and PCIe
contention. A row that was measured while a neighbour was running is flagged
rather than discarded, so the analysis can decide.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _nvidia_smi(query: str) -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001 -- provenance must never break a run
        return []


def _procs_on_gpu() -> int | None:
    """Compute processes currently on any visible GPU, this one included.

    None means we could not find out. That is deliberately different from 0:
    an unknown contention state must not read as a clean one, or a box where
    nvidia-smi is missing would silently certify all its timings.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            return None
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:  # noqa: BLE001
        return None


_STATIC: dict | None = None


def hardware_provenance() -> dict:
    """Everything needed to decide, later, whether two rows are comparable.

    The static half is queried once per process; only the contention count is
    re-read per row, because that is the only field that changes while a run is
    in flight. Three subprocess calls per row would be a real tax over a
    thousand-row grid.
    """
    global _STATIC
    if _STATIC is not None:
        return {**_STATIC, "concurrent_procs": _procs_on_gpu()}
    names = _nvidia_smi("name")
    drivers = _nvidia_smi("driver_version")
    prov = {
        "host": platform.node(),
        "gpu_name": names[0] if names else None,
        "gpu_count": len(names),
        "driver": drivers[0] if drivers else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # Number of compute processes on the box while this row was produced.
        # 1 means we were alone; more means wall_clock may carry contention.
        "concurrent_procs": _procs_on_gpu(),
    }
    try:
        import torch

        prov["torch"] = torch.__version__
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
            prov["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        prov["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    _STATIC = {k: v for k, v in prov.items() if k != "concurrent_procs"}
    return prov


def timing_is_trustworthy(prov: dict) -> bool:
    """False when another compute process shared the box. NFE and quality from
    such a row are still fine; its wall_clock is not."""
    n = prov.get("concurrent_procs")
    if n is None:
        return False   # unknown is not clean
    return n <= 1
