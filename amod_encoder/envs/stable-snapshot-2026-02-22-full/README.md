# Stable Snapshot — 2026-02-22 (Full)

**This is the FULLY-LOADED snapshot.** All three environments have every package
installed and verified: **60 pass, 0 fail, 1 skip** each (the skip is
`torch.compile` which requires Triton — Linux only).

## What's In Here

| File | Environment | PyTorch | Python | Tests |
|---|---|---|---|---|
| `torch-stable.yml` | General-purpose ML | 2.10.0+cu128 | 3.11 | 60/0/1 |
| `brain-encoders.yml` | Neuroimaging pipeline | 2.10.0+cu128 | 3.11 | 60/0/1 |
| `torch-nightly.yml` | Bleeding-edge testing | 2.12.0.dev+cu128 | 3.12 | 60/0/1 |
| `deep_compat_test.py` | 61-test compatibility suite | — | — | — |
| `README.md` | This file | — | — | — |

### Package Coverage (all envs)

Every env includes the full stack:

- **Core:** numpy, scipy, pandas, scikit-learn, matplotlib
- **PyTorch:** torch, torchvision, torchaudio (cu128, sm_120)
- **Models:** timm, transformers, accelerate, safetensors, huggingface-hub
- **Neuroimaging:** nibabel, nilearn, h5py
- **Stats:** statsmodels, pingouin
- **Media:** ffmpeg, opencv-python, Pillow (pip only!)
- **Data:** datasets, pydantic, pyyaml, typer
- **Monitoring:** nvidia-ml-py (not pynvml — deprecated), wandb
- **Dev:** pytest, rich, tqdm

## Installation

### torch-stable

```bash
conda env create -f torch-stable.yml
conda activate torch-stable
```

### brain-encoders

```bash
conda env create -f brain-encoders.yml
conda activate brain-encoders

# Install the amod_encoder package itself:
cd /path/to/amod_encoder
pip install -e ".[all]"
```

### torch-nightly (requires extra manual step!)

```bash
conda env create -f torch-nightly.yml
conda activate torch-nightly

# REQUIRED: conda YML can't pass --pre, so install torch manually:
pip install --pre --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

> **Why `--index-url` and not `--extra-index-url`?**
> With `--extra-index-url`, pip satisfies the torch requirement via CPU-only
> stable torch pulled as a transitive dep of timm/transformers from PyPI.
> `--index-url` forces the nightly index as primary, and `--force-reinstall`
> ensures it replaces whatever pip already resolved.

## Verify Installation

After creating any env, run the full compatibility suite:

```bash
python deep_compat_test.py
```

Expected output: **60 pass, 0 fail, 1 skip** (torch.compile — Triton is Linux-only).

The test covers:
1. Core imports (numpy, scipy, pandas, sklearn, matplotlib, PIL, yaml, etc.)
2. PyTorch CUDA (sm_120 arch, fp32/fp16/bf16 GPU matmul)
3. Torchvision + PIL DLL interaction (ToTensor, transforms.v2, JPEG/PNG/WebP)
4. Torchaudio (MelSpectrogram on CPU and GPU)
5. Timm model forward passes (ResNet-18, VGG-19 on GPU)
6. BLAS backend verification (must be OpenBLAS, not MKL)
7. Torch ↔ NumPy interop (CPU and CUDA roundtrips)
8. OpenMP DLL conflict detection (no duplicate runtimes)
9. Pillow install origin (must be pip, not conda)
10. Full ecosystem imports (transformers, wandb, nibabel, etc.)
11. torch.compile smoke test (Linux only — expected skip on Windows)
12. CUDA memory management (1 GB alloc/free cycle)

## Critical Windows Gotchas

### 1. Pillow: pip ONLY — never conda

Conda-installed Pillow ships imaging DLLs that conflict with pip-installed
torchvision on Windows → `Windows fatal exception: code 0xc0000138`.

**If you accidentally `conda install pillow`:**
```bash
conda remove pillow --force -y
pip install Pillow
```

The YMLs have comments marking this. Pillow gets installed automatically
as a pip dependency of torchvision.

### 2. BLAS: OpenBLAS ONLY — never MKL

Conda's default MKL ships Intel OpenMP (`libiomp5md.dll`), which conflicts
with PyTorch's bundled OpenMP (`libomp.dll`):
```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

All YMLs pin OpenBLAS to prevent this:
```yaml
- libblas=*=*openblas*
- libcblas=*=*openblas*
- liblapack=*=*openblas*
```

### 3. Blackwell (sm_120) requires cu128

RTX 5070 Ti uses Blackwell architecture (sm_120). Only PyTorch wheels from the
`cu128` index include sm_120 kernels. Older indexes won't work:

| Index | sm_120? | Works on 5070 Ti? |
|---|---|---|
| cu118 | No | No |
| cu124 | No | No |
| cu126 | No | No |
| **cu128** | **Yes** | **Yes** |

### 4. pynvml is deprecated

Use `nvidia-ml-py` instead. `pynvml` is an unmaintained wrapper that prints
deprecation warnings.

## Hardware

- **GPU:** NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120, 16 GB VRAM)
- **Driver:** 576.88 / CUDA 12.9
- **OS:** Windows 10/11
- **Conda:** miniconda3
