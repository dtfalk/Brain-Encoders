#!/usr/bin/env python
"""
Deep compatibility test for RTX 5070 Ti (Blackwell, sm_120) environments.

Tests every major import path, DLL integrity, GPU operations, and
cross-library interactions to catch sneaky conflicts early.

Usage:
    python envs/deep_compat_test.py
"""
from __future__ import annotations
import importlib
import sys
import os
import traceback
import time

# ── helpers ────────────────────────────────────────────────────────────────

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
SKIP = "\033[93m SKIP \033[0m"
SECT = "\033[96m"
RESET = "\033[0m"

results: list[tuple[str, str, str]] = []  # (section, test, status)


def _record(section: str, name: str, status: str, detail: str = ""):
    results.append((section, name, status))
    tag = PASS if status == "pass" else (FAIL if status == "fail" else SKIP)
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def section(title: str):
    print(f"\n{SECT}{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}{RESET}")


# ── 1. CORE IMPORTS ───────────────────────────────────────────────────────

section("1. Core Imports")

CORE_MODULES = [
    "numpy", "scipy", "pandas", "sklearn", "matplotlib",
    "PIL", "PIL.Image", "yaml", "requests", "tqdm", "rich",
]

for mod in CORE_MODULES:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", getattr(m, "VERSION", "ok"))
        _record("core_imports", mod, "pass", str(ver))
    except Exception as e:
        _record("core_imports", mod, "fail", str(e))


# ── 2. TORCH CORE ─────────────────────────────────────────────────────────

section("2. PyTorch Core")

try:
    import torch
    _record("torch_core", "import torch", "pass", torch.__version__)
except Exception as e:
    _record("torch_core", "import torch", "fail", str(e))
    print("\n  !! torch failed to import — skipping all GPU tests !!")
    torch = None

if torch is not None:
    # CUDA availability
    cuda_ok = torch.cuda.is_available()
    _record("torch_core", "torch.cuda.is_available()", "pass" if cuda_ok else "fail",
            str(cuda_ok))

    if cuda_ok:
        # Device info
        dev = torch.cuda.get_device_name(0)
        _record("torch_core", "GPU device name", "pass", dev)

        # CUDA version
        cuda_ver = torch.version.cuda
        _record("torch_core", "CUDA runtime version", "pass", str(cuda_ver))

        # sm_120 (Blackwell) arch check
        arch_list = torch.cuda.get_arch_list()
        has_sm120 = "sm_120" in arch_list or "sm_120a" in arch_list
        _record("torch_core", "sm_120 in arch list",
                "pass" if has_sm120 else "fail",
                ", ".join(arch_list[-5:]))

        # Basic tensor on GPU
        try:
            x = torch.randn(256, 256, device="cuda")
            y = torch.randn(256, 256, device="cuda")
            z = x @ y
            assert z.shape == (256, 256)
            _record("torch_core", "GPU matmul (256x256)", "pass")
            del x, y, z
        except Exception as e:
            _record("torch_core", "GPU matmul (256x256)", "fail", str(e))

        # Large matmul stress
        try:
            x = torch.randn(4096, 4096, device="cuda")
            y = torch.randn(4096, 4096, device="cuda")
            z = x @ y
            torch.cuda.synchronize()
            assert z.shape == (4096, 4096)
            _record("torch_core", "GPU matmul (4096x4096)", "pass")
            del x, y, z
        except Exception as e:
            _record("torch_core", "GPU matmul (4096x4096)", "fail", str(e))

        # Half precision (fp16)
        try:
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            z = x @ y
            torch.cuda.synchronize()
            _record("torch_core", "GPU fp16 matmul", "pass")
            del x, y, z
        except Exception as e:
            _record("torch_core", "GPU fp16 matmul", "fail", str(e))

        # BFloat16
        try:
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            z = x @ y
            torch.cuda.synchronize()
            _record("torch_core", "GPU bf16 matmul", "pass")
            del x, y, z
        except Exception as e:
            _record("torch_core", "GPU bf16 matmul", "fail", str(e))

        # VRAM allocation
        try:
            allocated = torch.cuda.memory_allocated(0) / 1e6
            reserved = torch.cuda.memory_reserved(0) / 1e6
            _record("torch_core", "VRAM reporting",
                    "pass", f"alloc={allocated:.0f}MB, reserved={reserved:.0f}MB")
        except Exception as e:
            _record("torch_core", "VRAM reporting", "fail", str(e))

        torch.cuda.empty_cache()


# ── 3. TORCHVISION + PIL INTERACTION ──────────────────────────────────────

section("3. Torchvision + PIL (DLL conflict zone)")

try:
    import torchvision
    _record("tv_pil", "import torchvision", "pass", torchvision.__version__)
except Exception as e:
    _record("tv_pil", "import torchvision", "fail", str(e))
    torchvision = None

if torchvision is not None:
    # The critical test: create an image with PIL, convert with torchvision
    try:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.new("RGB", (224, 224), color=(128, 64, 200))
        tensor = T.ToTensor()(img)
        assert tensor.shape == (3, 224, 224), f"wrong shape: {tensor.shape}"
        _record("tv_pil", "PIL→ToTensor pipeline", "pass")
    except Exception as e:
        _record("tv_pil", "PIL→ToTensor pipeline", "fail", str(e))

    # Torchvision transforms v2
    try:
        from torchvision.transforms import v2
        transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        out = transform(img)
        assert out.shape == (3, 224, 224)
        _record("tv_pil", "transforms.v2 pipeline", "pass")
    except Exception as e:
        _record("tv_pil", "transforms.v2 pipeline", "fail", str(e))

    # JPEG decoding (exercises libjpeg DLLs)
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        img2 = Image.open(buf)
        img2.load()
        _record("tv_pil", "PIL JPEG encode/decode", "pass")
    except Exception as e:
        _record("tv_pil", "PIL JPEG encode/decode", "fail", str(e))

    # PNG decoding
    try:
        from PIL import Image
        import io
        img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img2 = Image.open(buf)
        img2.load()
        _record("tv_pil", "PIL PNG encode/decode", "pass")
    except Exception as e:
        _record("tv_pil", "PIL PNG encode/decode", "fail", str(e))

    # WebP
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="WEBP")
        buf.seek(0)
        img2 = Image.open(buf)
        img2.load()
        _record("tv_pil", "PIL WebP encode/decode", "pass")
    except Exception as e:
        _record("tv_pil", "PIL WebP encode/decode", "fail", str(e))

    # Models on GPU
    if torch is not None and torch.cuda.is_available():
        try:
            model = torchvision.models.resnet18(weights=None)
            model = model.cuda().eval()
            x = torch.randn(1, 3, 224, 224, device="cuda")
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1000)
            _record("tv_pil", "resnet18 GPU forward", "pass")
            del model, x, out
            torch.cuda.empty_cache()
        except Exception as e:
            _record("tv_pil", "resnet18 GPU forward", "fail", str(e))


# ── 4. TORCHAUDIO ─────────────────────────────────────────────────────────

section("4. Torchaudio")

try:
    import torchaudio
    _record("torchaudio", "import torchaudio", "pass", torchaudio.__version__)
except ImportError:
    _record("torchaudio", "import torchaudio", "skip", "not installed")
    torchaudio = None
except Exception as e:
    _record("torchaudio", "import torchaudio", "fail", str(e))
    torchaudio = None

if torchaudio is not None and torch is not None:
    try:
        waveform = torch.randn(1, 16000)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80
        )(waveform)
        assert mel_spec.ndim == 3
        _record("torchaudio", "MelSpectrogram CPU", "pass",
                str(mel_spec.shape))
    except Exception as e:
        _record("torchaudio", "MelSpectrogram CPU", "fail", str(e))

    if torch.cuda.is_available():
        try:
            waveform = torch.randn(1, 16000, device="cuda")
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=80
            ).cuda()(waveform)
            torch.cuda.synchronize()
            _record("torchaudio", "MelSpectrogram GPU", "pass",
                    str(mel_spec.shape))
            del waveform, mel_spec
        except Exception as e:
            _record("torchaudio", "MelSpectrogram GPU", "fail", str(e))


# ── 5. TIMM ───────────────────────────────────────────────────────────────

section("5. Timm (model zoo)")

try:
    import timm
    _record("timm", "import timm", "pass", timm.__version__)
except Exception as e:
    _record("timm", "import timm", "fail", str(e))
    timm = None

if timm is not None and torch is not None:
    try:
        model = timm.create_model("resnet18", pretrained=False)
        _record("timm", "create_model resnet18", "pass")
    except Exception as e:
        _record("timm", "create_model resnet18", "fail", str(e))
        model = None

    if model is not None and torch.cuda.is_available():
        try:
            model = model.cuda().eval()
            x = torch.randn(1, 3, 224, 224, device="cuda")
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize()
            _record("timm", "resnet18 GPU forward", "pass", str(out.shape))
            del model, x, out
            torch.cuda.empty_cache()
        except Exception as e:
            _record("timm", "resnet18 GPU forward", "fail", str(e))

    # VGG19 (used in brain-encoders for FC7 features)
    try:
        model = timm.create_model("vgg19", pretrained=False)
        _record("timm", "create_model vgg19", "pass")
        if torch.cuda.is_available():
            model = model.cuda().eval()
            x = torch.randn(1, 3, 224, 224, device="cuda")
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize()
            _record("timm", "vgg19 GPU forward", "pass", str(out.shape))
            del model, x, out
            torch.cuda.empty_cache()
    except Exception as e:
        _record("timm", "vgg19 (create or GPU)", "fail", str(e))


# ── 6. NUMPY / SCIPY / BLAS SANITY ───────────────────────────────────────

section("6. NumPy / SciPy / BLAS")

try:
    import numpy as np
    # Check BLAS config — should be OpenBLAS, NOT MKL
    blas_info = np.show_config(mode="dicts") if hasattr(np, "show_config") else None
    if isinstance(blas_info, dict):
        blas_lib = blas_info.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    else:
        blas_lib = "check manually"
    _record("blas", "numpy BLAS backend", "pass", str(blas_lib))
except Exception as e:
    _record("blas", "numpy BLAS backend", "fail", str(e))

try:
    import numpy as np
    a = np.random.randn(2000, 2000)
    b = np.random.randn(2000, 2000)
    t0 = time.perf_counter()
    c = a @ b
    dt = time.perf_counter() - t0
    _record("blas", "numpy matmul (2000x2000)", "pass", f"{dt:.3f}s")
except Exception as e:
    _record("blas", "numpy matmul (2000x2000)", "fail", str(e))

try:
    from scipy import linalg
    import numpy as np
    a = np.random.randn(500, 500)
    u, s, vt = linalg.svd(a)
    assert u.shape == (500, 500)
    _record("blas", "scipy SVD (500x500)", "pass")
except Exception as e:
    _record("blas", "scipy SVD (500x500)", "fail", str(e))


# ── 7. TORCH ↔ NUMPY INTEROP ─────────────────────────────────────────────

section("7. Torch ↔ NumPy Interop")

if torch is not None:
    try:
        import numpy as np
        a = np.random.randn(1000, 1000).astype(np.float32)
        t = torch.from_numpy(a)
        assert t.shape == (1000, 1000)
        b = t.numpy()
        assert np.allclose(a, b)
        _record("interop", "numpy→torch→numpy roundtrip", "pass")
    except Exception as e:
        _record("interop", "numpy→torch→numpy roundtrip", "fail", str(e))

    if torch.cuda.is_available():
        try:
            a = np.random.randn(1000, 1000).astype(np.float32)
            t = torch.from_numpy(a).cuda()
            result = (t @ t.T).cpu().numpy()
            assert result.shape == (1000, 1000)
            _record("interop", "numpy→cuda→matmul→numpy", "pass")
        except Exception as e:
            _record("interop", "numpy→cuda→matmul→numpy", "fail", str(e))


# ── 8. OPENMP CONFLICT CHECK ─────────────────────────────────────────────

section("8. OpenMP Conflict Check")

# The test: import torch THEN numpy and do parallel operations
# If MKL's libiomp5md.dll and torch's libomp.dll clash, this crashes
try:
    if torch is not None:
        _ = torch.randn(100, 100)  # force torch OpenMP init
    import numpy as np
    # Force numpy to use multiple threads (BLAS call)
    a = np.random.randn(2000, 2000)
    b = a @ a.T
    _record("openmp", "torch + numpy coexistence", "pass")
except Exception as e:
    _record("openmp", "torch + numpy coexistence", "fail", str(e))

# Check for duplicate OpenMP DLLs (Windows-specific)
if sys.platform == "win32":
    env_prefix = sys.prefix
    dll_dir = os.path.join(env_prefix, "Library", "bin")
    lib_dir = os.path.join(env_prefix, "Lib", "site-packages", "torch", "lib")

    # Only flag if the SAME OpenMP library appears in both conda and torch dirs.
    # Different implementations (libgomp, vcomp140, libiomp5md) can coexist fine.
    # The real conflict: same DLL name in both → two copies loaded → crash.
    import re as _re
    _omp_pattern = _re.compile(r'^(lib[ig]?omp|vcomp|libomp).*\.dll$', _re.IGNORECASE)

    conda_omp = set()
    if os.path.isdir(dll_dir):
        conda_omp = {f.lower() for f in os.listdir(dll_dir) if _omp_pattern.match(f)}

    torch_omp = set()
    if os.path.isdir(lib_dir):
        torch_omp = {f.lower() for f in os.listdir(lib_dir) if _omp_pattern.match(f)}

    overlap = conda_omp & torch_omp
    if overlap:
        _record("openmp", "CONFLICTING OpenMP DLLs", "fail",
                f"same DLL in both dirs: {sorted(overlap)}")
    else:
        detail = (f"conda_bin: {sorted(conda_omp) or 'none'}, "
                  f"torch_lib: {sorted(torch_omp) or 'none'} (no overlap)")
        _record("openmp", "no conflicting OpenMP DLLs", "pass", detail)


# ── 9. PIL DLL ORIGIN CHECK ──────────────────────────────────────────────

section("9. Pillow DLL Origin")

try:
    import PIL
    pil_path = PIL.__file__
    is_conda = "conda" in pil_path.lower() and "site-packages" in pil_path.lower()
    # Check if it's a pip install by looking for dist-info
    pil_dir = os.path.dirname(os.path.dirname(pil_path))
    dist_infos = [d for d in os.listdir(pil_dir)
                  if d.startswith("pillow") and d.endswith(".dist-info")]
    is_pip = len(dist_infos) > 0

    if is_pip:
        _record("pil_dll", "Pillow installed via pip", "pass",
                f"v{PIL.__version__}")
    else:
        _record("pil_dll", "Pillow installed via pip", "fail",
                "appears to be conda-installed — risk of DLL conflict!")
except Exception as e:
    _record("pil_dll", "Pillow origin check", "fail", str(e))


# ── 10. OPTIONAL ECOSYSTEM ───────────────────────────────────────────────

section("10. Optional Ecosystem Imports")

OPTIONAL_MODULES = [
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers"),
    ("datasets", "datasets"),
    ("cv2", "opencv-python"),
    ("wandb", "wandb"),
    ("h5py", "h5py"),
    ("nibabel", "nibabel"),
    ("nilearn", "nilearn"),
    ("statsmodels", "statsmodels"),
    ("pydantic", "pydantic"),
    ("typer", "typer"),
    ("pingouin", "pingouin"),
]

for mod_name, pkg_name in OPTIONAL_MODULES:
    try:
        m = importlib.import_module(mod_name)
        ver = getattr(m, "__version__", getattr(m, "VERSION", "ok"))
        _record("ecosystem", f"{mod_name}", "pass", str(ver))
    except ImportError:
        _record("ecosystem", f"{mod_name}", "skip", "not installed")
    except Exception as e:
        _record("ecosystem", f"{mod_name}", "fail", str(e))


# ── 11. TORCH.COMPILE SMOKE TEST ─────────────────────────────────────────

section("11. torch.compile Smoke Test")

if sys.platform == "win32":
    _record("compile", "torch.compile basic", "skip",
            "Triton not available on Windows — Linux only")
elif torch is not None and torch.cuda.is_available():
    try:
        @torch.compile
        def f(x, y):
            return (x * y).sum()

        x = torch.randn(1000, device="cuda")
        y = torch.randn(1000, device="cuda")
        result = f(x, y)
        torch.cuda.synchronize()
        _record("compile", "torch.compile basic", "pass", str(result.item())[:8])
        del x, y, result
    except Exception as e:
        _record("compile", "torch.compile basic", "fail", str(e))
else:
    _record("compile", "torch.compile basic", "skip", "no CUDA")


# ── 12. CUDA MEMORY MANAGEMENT ───────────────────────────────────────────

section("12. CUDA Memory Management")

if torch is not None and torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Allocate ~1 GB
        big = torch.randn(256, 1024, 1024, device="cuda")  # ~1 GB fp32
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        _record("cuda_mem", "1GB allocation", "pass", f"peak={peak:.2f}GB")

        del big
        torch.cuda.empty_cache()

        after_free = torch.cuda.memory_allocated(0) / 1e6
        _record("cuda_mem", "memory freed after del+empty_cache", "pass",
                f"remaining={after_free:.0f}MB")
    except Exception as e:
        _record("cuda_mem", "memory management", "fail", str(e))


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════

section("SUMMARY")

n_pass = sum(1 for _, _, s in results if s == "pass")
n_fail = sum(1 for _, _, s in results if s == "fail")
n_skip = sum(1 for _, _, s in results if s == "skip")
total = len(results)

print(f"\n  Total: {total}  |  Pass: {n_pass}  |  Fail: {n_fail}  |  Skip: {n_skip}")

if n_fail > 0:
    print(f"\n  {FAIL} FAILURES:")
    for sec, name, status in results:
        if status == "fail":
            print(f"    - [{sec}] {name}")
    sys.exit(1)
else:
    print(f"\n  {PASS} ALL TESTS PASSED (or skipped as expected)")
    sys.exit(0)
