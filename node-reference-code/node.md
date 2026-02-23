# System & Environment Reference

Technical specification for the compute environment used by this project.

---

## Cluster: Midway3

| Detail | Value |
|--------|-------|
| **Cluster** | Midway3 |
| **Institution** | Research Computing Center (RCC), The University of Chicago |
| **Scheduler** | SLURM |
| **Login Nodes** | `midway3-login{1-4}.rcc.local` |
| **OS** | Red Hat Enterprise Linux 8.4 (Ootpa) |
| **Kernel** | 4.18.0-305.x86_64 |

### Documentation

- [RCC User Guide](https://rcc.uchicago.edu/docs/)
- [Midway3 Hardware](https://rcc.uchicago.edu/resources/midway3)
- [SLURM at RCC](https://rcc.uchicago.edu/docs/slurm/index.html)

---

## GPU Node

Our training and inference runs on a dedicated GPU node in the `hcn1-gpu` partition.

| Spec | Value |
|------|-------|
| **Node** | `midway3-0427` |
| **Partition** | `hcn1-gpu` |
| **Account** | `pi-hcn1` |
| **QOS** | `hcn1` (max walltime: 2 days) |

### GPUs

| Spec | Value |
|------|-------|
| **Model** | NVIDIA L40S |
| **Count** | 4 per node |
| **VRAM** | 48 GB GDDR6 per card (192 GB total) |
| **Architecture** | Ada Lovelace (AD102) |
| **CUDA Cores** | 18,176 per card |
| **Tensor Cores** | 568 per card (4th-gen) |
| **RT Cores** | 142 per card (3rd-gen) |
| **FP32** | 91.6 TFLOPS per card |
| **TF32 Tensor** | 183.2 TFLOPS per card |
| **FP16 Tensor** | 366.4 TFLOPS per card |
| **INT8 Tensor** | 733 TOPS per card |
| **Memory Bandwidth** | 864 GB/s per card |
| **TDP** | 350W per card |
| **PCIe** | Gen 4 x16 |
| **NVLink** | Not available (PCIe topology) |
| **Multi-GPU** | NCCL over PCIe |

> **Note:** The L40S is distinct from the L40 (no "S"). The L40S has higher TDP (350W vs 300W), higher clocks, and is optimized for AI/HPC workloads rather than professional visualization. It uses the full AD102 die.

### CPUs

| Spec | Value |
|------|-------|
| **Model** | Intel Xeon Gold 6346 |
| **Sockets** | 2 |
| **Cores per Socket** | 16 |
| **Total Cores** | 32 physical (HT disabled on compute) |
| **Clock** | 3.10 GHz base / 3.60 GHz turbo |
| **Architecture** | Ice Lake-SP (x86_64) |
| **L3 Cache** | 36 MB per socket |
| **TDP** | 205W per socket |

### Memory & Storage

| Spec | Value |
|------|-------|
| **System RAM** | 1 TB (1,031,735 MB) |
| **RAM Requested** | 200 GB (via `#SBATCH --mem=200G`) |
| **Storage Mount** | `/project` (Lustre parallel filesystem) |
| **Total Capacity** | 23 TB |
| **Available** | ~18 TB |

---

## SLURM Job Configuration

Our `submit.sh` requests:

```bash
#SBATCH --job-name=emnist-train-v2
#SBATCH --partition=hcn1-gpu
#SBATCH --account=pi-hcn1
#SBATCH --qos=hcn1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
```

| Resource | Requested | Available |
|----------|-----------|-----------|
| GPUs | 4 | 4 |
| CPUs | 32 | 32 |
| RAM | 200 GB | 1,007 GB |
| Wall Time | 8 hours | 2 days (QOS max) |

### Useful SLURM Commands

```bash
# Submit a job
sbatch submit.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <JOBID>

# View job details
scontrol show job <JOBID>

# View node details
scontrol show node midway3-0427

# View partition info
sinfo -p hcn1-gpu

# Check past job stats (after completion)
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,TotalCPU,AllocGRES
```

---

## Software Stack

### Module Environment

```bash
module load cuda/12.6
module load python/miniforge-25.3.0
```

### Conda Environment

| Component | Version |
|-----------|---------|
| **Environment Name** | `superstition-sd` |
| **Python** | 3.11.14 |
| **PyTorch** | 2.9.0+cu126 |
| **CUDA Toolkit** | 12.6 |
| **cuDNN** | 9.10.2 (build 91002) |
| **NCCL** | 2.27.5 (via nvidia-nccl-cu12) |
| **diffusers** | 0.36.0 |
| **torchvision** | 0.24.0+cu126 |
| **numpy** | 2.4.1 |
| **matplotlib** | 3.10.8 |
| **Pillow** | 12.1.0 |
| **rich** | 14.3.1 |
| **pynvml / nvidia-ml-py** | 13.x |
| **ffmpeg** | 8.0.1 (conda-forge) |
| **accelerate** | 1.12.0 |
| **transformers** | 4.57.6 |
| **huggingface-hub** | 0.34.4 |

### Reproducing the Environment

A full environment export is saved at `environment.yml` in the project root.

```bash
# Create from export (exact reproduction)
conda env create -f environment.yml

# Or create manually with core packages
conda create -n superstition-sd python=3.11
conda activate superstition-sd
conda install -c conda-forge ffmpeg pillow rich
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install diffusers accelerate transformers
pip install matplotlib numpy pynvml
```

### Environment Variables

Set in `training/submit.sh` before training:

```bash
export TORCH_HOME="${PROJECT_ROOT}/data"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Variable | Purpose |
|----------|---------||
| `TORCH_HOME` | EMNIST dataset cache (`data/` inside the project) |
| `PYTHONUNBUFFERED` | Force unbuffered stdout/stderr so SLURM logs update in real-time |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Enables CUDA memory allocator to grow segments dynamically, reducing fragmentation on L40S |

---

## Distributed Training Setup

| Detail | Value |
|--------|-------|
| **Framework** | PyTorch DistributedDataParallel (DDP) |
| **Launch** | `torchrun --standalone --nproc_per_node=4` |
| **Backend** | NCCL (GPU-to-GPU gradient sync) |
| **Topology** | Single-node, 4 GPUs over PCIe |
| **Gradient Sync** | All-reduce via NCCL ring |
| **Mixed Precision** | FP16 with `torch.cuda.amp.GradScaler` |

### How DDP Works on This Node

```
                    ┌──────────────────────────────────────┐
                    │         midway3-0427 (1 node)        │
                    │                                      │
                    │  ┌────────┐  ┌────────┐              │
                    │  │ GPU 0  │  │ GPU 1  │              │
                    │  │ L40S   │  │ L40S   │              │
                    │  │ 48GB   │  │ 48GB   │              │
                    │  └───┬────┘  └───┬────┘              │
                    │      │  PCIe     │                   │
                    │      │  ◄────►   │                   │
                    │      │  NCCL     │                   │
                    │      │  ◄────►   │                   │
                    │  ┌───┴────┐  ┌───┴────┐              │
                    │  │ GPU 2  │  │ GPU 3  │              │
                    │  │ L40S   │  │ L40S   │              │
                    │  │ 48GB   │  │ 48GB   │              │
                    │  └────────┘  └────────┘              │
                    │                                      │
                    │  CPU: 2× Xeon Gold 6346 (32 cores)   │
                    │  RAM: 1 TB                           │
                    └──────────────────────────────────────┘
```

Each GPU runs its own process (rank 0–3). The dataset is sharded via `DistributedSampler`, and gradients are synchronized after each backward pass using NCCL all-reduce over PCIe.

---

## File System Layout

The project is entirely self-contained:

```
emnist-ddpm/                       # ← Project root
├── environment.yml                # Conda env export
├── SYSTEM.md                      # This file
├── README.md                      # Project overview
├── training/                      # Training pipeline
│   ├── train.py
│   ├── submit.sh
│   ├── pretty_logger.py
│   ├── metrics/
│   └── logs/
├── inference/                     # Inference engine
│   ├── run_inference.py
│   ├── config.py
│   ├── metrics/
│   ├── utils/
│   └── zoos/
├── checkpoints/                   # Trained model weights
│   ├── small/                     # 28×28 models
│   ├── medium/                    # 64×64 models
│   └── large/                     # 96×96 models
├── data/                          # EMNIST dataset (auto-downloaded)
└── output/                        # Generated videos/frames
```

---

## Performance Notes

### L40S for Diffusion Training

The NVIDIA L40S is well-suited for this workload:

- **48 GB VRAM** per card handles the UNet2DModel comfortably at all size presets (small through XL)
- **4th-gen Tensor Cores** accelerate FP16 mixed-precision training (366 TFLOPS)
- **864 GB/s memory bandwidth** keeps the UNet's many convolution layers fed
- **PCIe Gen 4** interconnect is sufficient for single-node DDP with the relatively small EMNIST model (gradient payloads are modest)

### Expected Throughput

| Size Preset | Resolution | Batch/GPU | ~Samples/sec (4 GPU) | ~Epoch Time |
|-------------|-----------|-----------|---------------------|-------------|
| `small` | 28×28 | 2048 | ~15,000–20,000 | ~7s |
| `medium` | 64×64 | 384 | ~2,500–4,000 | ~30s |
| `large` | 96×96 | 96 | ~400–800 | ~2–4min |
| `xl` | 128×128 | 32 | ~100–200 | ~8–15min |

> These are rough estimates. Actual throughput depends on system load, PCIe contention, and thermal throttling. The training dashboard shows live samples/sec.

### Memory Usage

| Size | VRAM/GPU (approx) | System RAM |
|------|-------------------|------------|
| `small` | ~4–6 GB | ~30 GB |
| `medium` | ~8–12 GB | ~40 GB |
| `large` | ~18–24 GB | ~60 GB |
| `xl` | ~30–40 GB | ~80 GB |

---

*APEX Laboratory — The University of Chicago*
