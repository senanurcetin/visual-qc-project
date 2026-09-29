"""Pre-flight checks for a real (GPU) run of run_dl_case_study.py.

`evaluate` is a pure function over a dict of facts, so the decision logic is unit tested; the
probing that builds those facts (`collect_facts`) is the only part that touches the machine.
"""
from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXPECTED_CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
EXPECTED_PER_CLASS = 300
MIN_FREE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Check:
    status: str  # PASS | WARN | FAIL
    name: str
    detail: str


def label_of(filename: str) -> str:
    """NEU-CLS file names look like `pitted_surface_12.jpg`."""
    return Path(filename).stem.rsplit("_", 1)[0]


def check_dataset_layout(filenames: list[str]) -> Check:
    counts = Counter(label_of(f) for f in filenames)
    expected = {c: EXPECTED_PER_CLASS for c in EXPECTED_CLASSES}
    if dict(counts) == expected:
        return Check("PASS", "dataset layout", f"{len(filenames)} images, {EXPECTED_PER_CLASS} per class in {len(EXPECTED_CLASSES)} classes")
    problems = [f"{c}: found {counts.get(c, 0)}, expected {n}" for c, n in expected.items() if counts.get(c, 0) != n]
    problems += [f"unexpected class '{c}' ({n} files)" for c, n in counts.items() if c not in expected]
    return Check("FAIL", "dataset layout", "; ".join(problems))


def evaluate(facts: dict) -> list[Check]:
    checks: list[Check] = []

    if facts.get("torch_error"):
        checks.append(Check("FAIL", "torch", f"cannot import torch/torchvision: {facts['torch_error']} (pip install -r requirements-dl.txt)"))
    else:
        checks.append(Check("PASS", "torch", f"torch {facts['torch_version']}"))
        if facts.get("cuda_device"):
            gb = facts.get("cuda_memory_gb")
            checks.append(Check("PASS", "gpu", f"{facts['cuda_device']}" + (f", {gb:.0f} GB" if gb else "")))
        else:
            checks.append(Check("WARN", "gpu", "no CUDA device found: a real run on CPU will take hours (install a CUDA build of torch)"))

    free = facts.get("free_bytes", 0)
    if free >= MIN_FREE_BYTES:
        checks.append(Check("PASS", "disk", f"{free / 1024**3:.1f} GB free"))
    else:
        checks.append(Check("FAIL", "disk", f"{free / 1024**3:.1f} GB free, need at least {MIN_FREE_BYTES / 1024**3:.0f} GB"))

    if facts.get("dataset_files") is not None:
        checks.append(check_dataset_layout(facts["dataset_files"]))
        unreadable = facts.get("unreadable_samples", [])
        if unreadable:
            checks.append(Check("FAIL", "dataset images", f"cannot decode: {', '.join(unreadable[:3])}"))
    elif facts.get("zip_present"):
        checks.append(Check("WARN", "dataset", "archive downloaded but not extracted yet; the run will extract it"))
    elif facts.get("dataset_url_ok"):
        checks.append(Check("PASS", "dataset", "not downloaded yet, but the download URL is reachable"))
    else:
        checks.append(Check(
            "FAIL", "dataset",
            f"not on disk and the download is unreachable ({facts.get('dataset_url_detail', 'no detail')}); "
            f"download NEU-CLS manually and place the zip at {facts.get('zip_path')}",
        ))

    if facts.get("weights_cached"):
        checks.append(Check("PASS", "pretrained weights", "ImageNet weights found in the torch cache"))
    elif facts.get("weights_url_ok"):
        checks.append(Check("PASS", "pretrained weights", "will be downloaded on first use (host reachable)"))
    else:
        checks.append(Check("WARN", "pretrained weights", "not cached and the host is unreachable: pre-download them or pass --no-pretrained"))

    checks.append(
        Check("PASS", "baseline file", "Random Forest results found for the comparison")
        if facts.get("baseline_present")
        else Check("FAIL", "baseline file", "docs/data/neu-cls-case-study/benchmark-comparison.json is missing")
    )
    checks.append(
        Check("PASS", "output directories", "writable")
        if facts.get("outputs_writable")
        else Check("FAIL", "output directories", "cannot write to docs/data/neu-cls-dl or the cache directory")
    )
    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "FAIL" for c in checks) else 0


def render(checks: list[Check]) -> str:
    lines = [f"[{c.status:<4}] {c.name}: {c.detail}" for c in checks]
    verdict = "NOT ready" if exit_code(checks) else "ready" + (" (with warnings)" if any(c.status == "WARN" for c in checks) else "")
    return "\n".join([*lines, "", f"Preflight: {verdict}"])


def probe_url(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Is the host reachable and willing to serve this URL? (HEAD; a redirect counts as reachable.)"""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 400, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - any network failure means "not reachable"
        return False, type(exc).__name__


def collect_facts(*, cache_dir: Path, zip_path: Path, image_dirs: list[Path], dataset_url: str, baseline_file: Path,
                  output_dir: Path, weights_url: str = "https://download.pytorch.org/models/resnet18-f37072fd.pth") -> dict:
    facts: dict = {"zip_path": str(zip_path)}
    try:
        import torch
        import torchvision  # noqa: F401

        facts["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            facts["cuda_device"], facts["cuda_memory_gb"] = props.name, props.total_memory / 1024**3
        cached = Path(torch.hub.get_dir()) / "checkpoints"
        facts["weights_cached"] = cached.is_dir() and any(cached.glob("resnet18-*.pth"))
    except Exception as exc:  # noqa: BLE001
        facts["torch_error"] = f"{type(exc).__name__}: {exc}"
    if not facts.get("weights_cached"):
        facts["weights_url_ok"], _ = probe_url(weights_url)

    probe_dir = cache_dir if cache_dir.exists() else cache_dir.parent
    facts["free_bytes"] = shutil.disk_usage(probe_dir).free

    if all(d.is_dir() for d in image_dirs):
        files = sorted(f for d in image_dirs for f in d.glob("*.jpg"))
        facts["dataset_files"] = [f.name for f in files]
        try:
            import cv2

            facts["unreadable_samples"] = [f.name for f in files[:: max(1, len(files) // 5)] if cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) is None]
        except ImportError:
            pass
    else:
        facts["zip_present"] = zip_path.exists()
        if not facts["zip_present"]:
            facts["dataset_url_ok"], facts["dataset_url_detail"] = probe_url(dataset_url)

    facts["baseline_present"] = baseline_file.exists()
    try:
        for d in (output_dir, cache_dir):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("x")
            probe.unlink()
        facts["outputs_writable"] = True
    except OSError:
        facts["outputs_writable"] = False
    return facts
