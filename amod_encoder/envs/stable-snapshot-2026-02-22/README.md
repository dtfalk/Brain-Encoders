# Stable Snapshot — 2026-02-22 (Original)

**This is the ORIGINAL snapshot** taken before adding the full package set.
These YMLs have the minimal packages per environment (brain-encoders had no
torchaudio, torch-stable had no neuroimaging packages, etc.).

For the fully-loaded version, see `stable-snapshot-2026-02-22-full/`.

## What's In Here

| File | Environment | PyTorch | Python |
|---|---|---|---|
| `torch-stable.yml` | General-purpose ML | 2.10.0+cu128 | 3.11 |
| `brain-encoders.yml` | Neuroimaging pipeline | 2.10.0+cu128 | 3.11 |
| `torch-nightly.yml` | Bleeding-edge testing | nightly+cu128 | 3.12 |
| `environment-local.yml` | Legacy (from prior session) | 2.10.0+cu128 | 3.11 |
| `environment-node.yml` | HPC node config | — | — |
| `deep_compat_test.py` | 61-test compatibility suite | — | — |

## Installation

### torch-stable or brain-encoders

```bash
conda env create -f torch-stable.yml
conda activate torch-stable
```

For brain-encoders, also install the package:
```bash
conda activate brain-encoders
pip install -e ".[all]"
```

### torch-nightly (requires extra step)

```bash
conda env create -f torch-nightly.yml
conda activate torch-nightly

# Conda YML can't do --pre, so install torch manually:
pip install --pre --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Use `--index-url`, NOT `--extra-index-url`** — otherwise pip pulls CPU-only
stable torch as a transitive dependency of timm/transformers.

## Critical Windows Gotchas

### 1. NEVER install Pillow via conda

Conda Pillow DLLs conflict with pip torchvision → `fatal exception: code 0xc0000138`.

If you accidentally do it:
```bash
conda remove pillow --force -y
pip install Pillow
```

### 2. NEVER use MKL BLAS

Conda's default MKL ships `libiomp5md.dll` which conflicts with torch's bundled
`libomp.dll` → crash on import. All YMLs pin OpenBLAS:
```yaml
- libblas=*=*openblas*
- libcblas=*=*openblas*
- liblapack=*=*openblas*
```

### 3. Blackwell (sm_120) needs cu128

Only `cu128` pip wheels include sm_120 kernels. Older indexes (cu118/cu124/cu126) won't work on RTX 5070 Ti.

### 4. Verify after install

```bash
python deep_compat_test.py
```

Should show 0 failures.

## Hardware

- **GPU:** NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120, 16 GB VRAM)
- **Driver:** 576.88 / CUDA 12.9
- **OS:** Windows
