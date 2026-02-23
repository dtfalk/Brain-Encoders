# Zoos — Force & Drift Registries

Composable dynamics modules for steering the diffusion generation process. Forces operate in **score space** (modifying the UNet's predicted noise), while drifts operate in **state space** (modifying the latent sample after each scheduler step).

---

## Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| [`force_zoo.py`](force_zoo.py) | Score-space force registry | `compose_forces()`, `FORCE_REGISTRY` |
| [`drift_zoo.py`](drift_zoo.py) | State-space drift registry | `resolve_drift_fn()`, `DRIFT_REGISTRY`, `drift_none()` |
| [`presets.py`](presets.py) | Ready-made force stack configurations | `attract_class()`, `subtract()`, `blend()`, `oscillate()`, … |
| [`__init__.py`](__init__.py) | Package exports | — |

---

## Conceptual Framework

```
┌─────────────────────────────────────────────────────┐
│                 Diffusion Step t                    │
│                                                     │
│  1. ε = UNet(x_t, t, class_label)     ← raw score  │
│                                                     │
│  2. ε' = compose_forces(ε, stack, t)  ← FORCES     │
│     ├── attract(target)                             │
│     ├── repel(distractor)                           │
│     ├── blend(A, B, weights)                        │
│     └── oscillate(A, B, freq)                       │
│                                                     │
│  3. x_{t-1} = scheduler.step(ε', t, x_t)           │
│                                                     │
│  4. x_{t-1} = drift(x_{t-1}, t, cfg)  ← DRIFTS    │
│     ├── drift_none (identity)                       │
│     ├── drift_noise (thermal exploration)           │
│     ├── drift_decay (pull toward zero)              │
│     └── drift_cosine (time-varying)                 │
└─────────────────────────────────────────────────────┘
```

### Score Space vs State Space

| Dimension | Score Space (Forces) | State Space (Drifts) |
|-----------|---------------------|---------------------|
| **What it modifies** | Predicted noise ε_θ | Latent sample x_t |
| **When applied** | Before scheduler step | After scheduler step |
| **Analogy** | Changing the wind direction | Pushing the boat directly |
| **Use cases** | Class guidance, repulsion, blending | Exploration noise, decay, custom dynamics |

---

## Force Zoo (`force_zoo.py`)

All forces follow a consistent signature and are registered in `FORCE_REGISTRY`:

```python
def force_fn(x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state):
    → Optional[torch.Tensor]  # additive term on the score
```

### Available Forces

| Force | Description | Key Config |
|-------|-------------|------------|
| `attract` | Pull toward a class basin | `target`, `strength` |
| `repel` | Push away from a class basin | `target`, `strength` |
| `blend` | Weighted mix of class attractions | `targets`, `weights` |
| `subtract` | Attract to one, repel from another | `pos_class`, `neg_class`, `w_pos`, `w_neg` |
| `oscillate` | Time-varying oscillation between classes | `class_a`, `class_b`, `frequency` |
| `drift_noise` | Additive Gaussian noise on score | `sigma` |

---

## Drift Zoo (`drift_zoo.py`)

Drifts modify x after the scheduler step:

```python
def drift_fn(x, t, step_index, total_steps, cfg):
    → torch.Tensor  # modified state
```

### Available Drifts

| Drift | Description | Key Config |
|-------|-------------|------------|
| `drift_none` | Identity (no-op) | — |
| `drift_noise` | Gaussian perturbation | `sigma` |
| `drift_decay` | Exponential decay toward zero | `rate` |
| `drift_cosine` | Time-varying cosine modulation | `amplitude`, `frequency` |

---

## Presets (`presets.py`)

Convenience functions that return ready-made `force_stack` lists:

```python
from inference.zoos.presets import subtract

config = InferenceConfig(
    character="k",
    force=ForceConfig(
        force_stack=subtract("k", "m", w_pos=3.0, w_neg=2.0),
    ),
)
```

### Available Presets

| Preset | Behavior | Params |
|--------|----------|--------|
| `null_wander(sigma)` | Random walk without class attraction | `sigma` |
| `attract_class(target, strength)` | Pure CFG-style class attraction | `target`, `strength` |
| `subtract(pos, neg, w_pos, w_neg)` | Attract + repel (contrastive) | classes, weights |
| `blend(targets, weights)` | Multi-class weighted blend | lists |
| `oscillate(a, b, freq)` | Sinusoidal oscillation between classes | classes, frequency |

---

## Energy Landscape Interpretation

Each EMNIST class (A, B, …, z, 0–9) defines a **basin** in the score manifold. The unconditional prediction is the "background field":

- **Attract** = positive gravity toward a basin
- **Repel** = negative gravity away from a basin
- **Blend** = superposition of gravitational fields
- **Subtract** = contrastive guidance (maximize one class, minimize another)
- **Oscillate** = the trajectory swings between basins over time
