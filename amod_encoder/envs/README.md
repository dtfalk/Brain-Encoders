# Environment Setup — Brain-Encoders (RTX 5070 Ti)

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) |
| VRAM | 16 GB GDDR7 |
| Driver | 576.88+ |
| CUDA (driver) | 12.9 |
| CUDA (PyTorch) | 12.8 (cu128 wheels — includes sm_120 kernels) |

## Three Environments

| Environment | Python | PyTorch | Purpose |
|---|---|---|---|
| **torch-stable** | 3.11 | 2.10.0+cu128 (stable) | General-purpose: WhisperX, NeMo, diffusion, fine-tuning, inference |
| **brain-encoders** | 3.11 | 2.10.0+cu128 (stable) | Neuroimaging encoding-model pipeline (this repo) |
| **torch-nightly** | 3.12 | nightly+cu128 | Bleeding-edge testing, torch.compile experiments |

### Quick Start

```bash
# Create any environment
conda env create -f envs/<name>.yml
conda activate <name>

# For brain-encoders, also install the package
pip install -e ".[all]"

# For torch-nightly, install torch manually after env creation
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Verify GPU

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Full Compatibility Test

```bash
python envs/deep_compat_test.py
```

Runs 61 tests covering: core imports, CUDA/sm_120 arch, GPU matmul (fp32/fp16/bf16), torchvision+PIL pipelines, torchaudio, timm model forward passes, BLAS backend verification, torch↔numpy interop, OpenMP DLL conflict detection, Pillow install origin, ecosystem imports, and CUDA memory management.

## Critical Windows Gotchas

These are hard-won lessons from debugging on this machine. **Do not skip these.**

### 1. Pillow: pip only, NEVER conda

Conda-installed Pillow ships DLLs that conflict with pip-installed torchvision on Windows, causing `fatal exception: code 0xc0000138`. The YML files have comments marking this.

**If you accidentally `conda install pillow`:**
```bash
conda remove pillow --force -y
pip install Pillow
```

### 2. BLAS: OpenBLAS only, NEVER MKL

Conda's default MKL BLAS ships `libiomp5md.dll` (Intel OpenMP), which conflicts with PyTorch's bundled `libomp.dll`, causing:
```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

All YML files pin OpenBLAS:
```yaml
- libblas=*=*openblas*
- libcblas=*=*openblas*
- liblapack=*=*openblas*
```

### 3. Blackwell (sm_120) requires cu128

Older CUDA wheel indexes (cu118, cu124, cu126) do **not** include sm_120 kernels. Always use:
```
--extra-index-url https://download.pytorch.org/whl/cu128        # stable
--index-url https://download.pytorch.org/whl/nightly/cu128      # nightly
```

### 4. torch-nightly: use `--index-url`, not `--extra-index-url`

With `--extra-index-url`, pip may pull CPU-only stable torch as a dependency of timm/transformers instead of the nightly. Use `--index-url` with `--force-reinstall`:
```bash
pip install --pre --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 5. pynvml is deprecated → use nvidia-ml-py

`pynvml` is a deprecated wrapper around `nvidia-ml-py`. Use `nvidia-ml-py>=12.0` in new environments.

## Snapshot / Rollback

A frozen copy of all YML files from the first verified-clean state is saved in:
```
envs/stable-snapshot-2026-02-22/
```

To roll back to that state:
```bash
conda env remove -n <name> -y
conda env create -f envs/stable-snapshot-2026-02-22/<name>.yml
```

## Installed Packages per Environment

<details>
<summary><strong>torch-stable</strong> — full-stack ML</summary>

**Core:** numpy, scipy, pandas, scikit-learn, matplotlib  
**PyTorch:** torch, torchvision, torchaudio (cu128)  
**Models:** timm, transformers, accelerate, safetensors, tokenizers, datasets, huggingface-hub  
**Media:** ffmpeg, opencv-python, torchcodec  
**Dev:** jupyterlab, ipywidgets, pytest  
**Monitoring:** nvidia-ml-py, wandb  
**Utilities:** rich, tqdm, pyyaml, requests
</details>

<details>
<summary><strong>brain-encoders</strong> — neuroimaging pipeline</summary>

**Core:** numpy, scipy, pandas, scikit-learn, matplotlib  
**PyTorch:** torch, torchvision, torchaudio (cu128)  
**Neuroimaging:** nibabel, nilearn  
**Models:** timm, transformers, opencv-python  
**Stats:** statsmodels, pingouin  
**Pipeline:** pyyaml, typer, h5py, pydantic  
**Dev:** pytest, pytest-cov, ruff, mypy  
**Monitoring:** nvidia-ml-py  
**Utilities:** rich, ffmpeg
</details>

<details>
<summary><strong>torch-nightly</strong> — bleeding edge</summary>

**Core:** numpy, scipy, pandas, scikit-learn, matplotlib  
**PyTorch:** torch, torchvision, torchaudio (nightly cu128) — installed manually  
**Models:** timm, transformers, accelerate, safetensors, huggingface-hub  
**Media:** ffmpeg, opencv-python  
**Dev:** jupyterlab, pytest  
**Monitoring:** nvidia-ml-py  
**Utilities:** rich
</details>
