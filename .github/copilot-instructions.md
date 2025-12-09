# AI Coding Agent Instructions for openpi

## Project Overview

**openpi** is an open-source robotics VLA (Vision-Language-Action) model framework by Physical Intelligence. It contains three main model variants:
- **π₀**: Flow-based VLA with superior generalization
- **π₀-FAST**: Autoregressive variant using the FAST action tokenizer
- **π₀.₅**: Upgraded π₀ with knowledge insulation for better open-world generalization

The repo supports both JAX and PyTorch implementations, multiple robot platforms (ALOHA, DROID, LIBERO), and comprehensive training/inference pipelines.

## Architecture & Key Components

### Core Model Structure (`src/openpi/models/`)

**Model System**:
- `model.py`: Defines `Observation`, `Actions`, and `BaseModel` interface. Key concept: **data is nested dicts → structured dataclasses**
  - `Observation.from_dict()` converts raw data dict to structured format
  - Images always 3-channel RGB in [-1, 1] float32
  - State is low-dimensional robot joint/end-effector data
  - Tokenized prompts are optional (language conditioning)

**Model Implementations**:
- `pi0.py`: Flow-matching head, multi-camera vision encoding with SigLIP, action generation via action expert
- `pi0_fast.py`: Autoregressive GPT-style action prediction with token-level AR masking
- `gemma.py`/`gemma_fast.py`: Language model backbones (Gemma 2B variant)
- `siglip.py`: Vision encoder (SigLIP So400m/14)
- `tokenizer.py`: SentencePiece-based prompt tokenization

**Dual Framework Support**:
- JAX models use Flax NNX (Neural Next-gen JAX)
- PyTorch models in `models_pytorch/` with weight conversion utilities
- Weights are loaded via `orbax.checkpoint` or SafeTensors

### Training Pipeline (`src/openpi/training/`)

**Configuration System** (`config.py`):
- Massive hierarchical config using `tyro` CLI generation
- Key dataclasses: `TrainConfig`, `DataConfig`, `AssetsConfig`, `ModelConfig`
- Configs include: optimizer, LR schedule, data loading, sharding, checkpoint paths
- Pre-configured named configs (e.g., `pi0_droid`, `pi05_libero`) accessible via `get_config()`

**Data System**:
- `data_loader.py`: Protocol-based DataLoader/Dataset interfaces
- Supports LeRobot datasets, RLDS format (DROID), and custom formats
- **Transform pipeline**: repack → normalize → model-specific transforms
- Data fetching via `TransformedDataset` (applies transforms on-the-fly)
- Normalization stats cached in checkpoint `assets/` dir for reproducibility

**Training Loop** (`scripts/train.py`):
- JAX-based distributed training with Flax NNX and FSDP sharding
- `train_step()` computes loss and optimizer updates
- Weight loading from base model checkpoints
- W&B integration for experiment tracking
- Checkpoint saving via Orbax with resume capability

### Data Transforms (`src/openpi/transforms.py`)

**Key Concept**: Transforms are composable functions (protocol-based duck typing)
- `RepackTransform`: Dict-to-dict mapping (e.g., `observation.image.base` → `images.base_0_rgb`)
- `Normalize`: Z-score or quantile normalization (uses precomputed stats)
- `InjectDefaultPrompt`: Adds default language instruction
- `Group`: Organizes input/output transforms
- Custom transforms extend `DataTransformFn` protocol

### Policy & Inference (`src/openpi/policies/`)

**Policy Abstractions**:
- `policy.py`: Base policy interface with `infer()` method
- `droid_policy.py`, `aloha_policy.py`, `libero_policy.py`: Robot-specific policies
- Policies wrap models, apply correct preprocessing, handle tensor conversions

**Serving** (`src/openpi/serving/`):
- WebSocket server for real-time inference
- Checkpoint loading and model compilation
- Latency optimized for robotics (target ~10-30ms per inference)

## Critical Developer Workflows

### 1. **Installation & Setup**
```bash
# Clone with submodules (ALOHA, LIBERO are third_party/*)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Use uv (required Python dependency manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Key Environment Variable**:
- `OPENPI_DATA_HOME` (default `~/.cache/openpi`): Where base checkpoints download
- `GIT_LFS_SKIP_SMUDGE=1` prevents Git LFS smudging (LeRobot is heavy)

### 2. **Running Training**
```bash
# Training uses config-driven approach via tyro CLI:
python scripts/train.py --config=pi0_droid_config
# Or with overrides:
python scripts/train.py --config=pi0_droid --batch-size=4 --learning-rate=1e-4

# Config specifies: model, data, optimizer, checkpoint paths, W&B settings
# Checkpoints auto-save; resume with: --resume
```

**Key Files to Understand**:
- `src/openpi/training/config.py`: Lists all named configs in `_CONFIGS` dict
- `scripts/train.py` `main()`: Entry point, handles setup and training loop

### 3. **Running Inference**
```bash
# Programmatic (most common):
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
result = policy.infer(example)  # Example: dict with images, state, optional prompt

# See examples/inference.ipynb for full patterns
```

### 4. **Running Tests**
```bash
# Pytest with custom JAX CPU fallback (conftest.py):
pytest src/openpi/
pytest src/openpi/transforms_test.py::test_delta_actions  # Specific test

# Tests auto-detect GPU; use CPU if unavailable (set JAX_PLATFORMS=cpu)
# Fixtures: tmp_path (pytest), custom mocking patterns in download_test.py
```

### 5. **Code Quality**
```bash
# Required pre-commit hooks:
pre-commit install
pre-commit run --all-files

# Linting:
ruff check src/
ruff format src/  # Line length = 120 chars
ruff check --fix src/

# Type checking: Uses `@at.typecheck` decorator (jaxtyping + beartype)
```

## Project-Specific Conventions

### 1. **Array Type Annotations** (`src/openpi/shared/array_typing.py`)
- **Always use** `@at.typecheck` decorator for functions accepting arrays
- Example: `@at.typecheck def foo(x: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b"]`
- Supports JAX arrays, PyTorch tensors, NumPy arrays (via union `Array`)
- Runtime type checking via beartype; disable with `at.disable_typechecking()` context manager
- Patch for JAX tracing: Skip typechecking when `tree_util` in stack (avoids spurious errors during JIT)

### 2. **Data Structure Conventions**
- **Data dicts** are always nested dicts with string keys, numpy/JAX/torch leaves
- **Observation format** (see `model.py`):
  ```python
  {
    "image": {"base_0_rgb": float32[*b,h,w,3], "left_wrist_0_rgb": ..., ...},
    "image_mask": {"base_0_rgb": bool[*b], ...},
    "state": float32[*b, state_dim],
    "tokenized_prompt": int32[*b, seq_len],  # optional
    "tokenized_prompt_mask": bool[*b, seq_len],  # optional
    "token_ar_mask": int32[*b, seq_len],  # π0-FAST only
    "token_loss_mask": bool[*b, seq_len],  # π0-FAST only
  }
  ```
- **Transform output** must match this structure exactly

### 3. **Config System Patterns**
- Configs are **frozen dataclasses** (immutable)
- Use `dataclasses.replace()` for config overrides, not mutation
- Named configs live in `_CONFIGS` dict in `config.py`; add via `register_config()`
- Configs deterministically specify all model/training parameters (reproducibility)

### 4. **Flax NNX vs. JAX**
- Models use **Flax NNX** (successor to Flax linen), not pure JAX
- `nnx.Module` subclasses define model architecture
- `nnx.split(model)` → `(graphdef, state)` for serialization
- `nnx.merge()` reconstructs model; patterns: `nnx.Rngs`, `nnx.Dict`, `nnx.Linear`, `nnx.Embed`
- JAX operations wrapped in NNX for functional training (vs. pure JAX functional style)

### 5. **Model Weight Loading**
- Base checkpoints from `gs://openpi-assets/checkpoints/{model_name}`
- Weights in Orbax `.orbax` dir or SafeTensors format
- `weight_loaders.py`: Flexible weight assignment via path patterns
- LoRA fine-tuning: `lora.py` defines low-rank adapter layers
- Always validate weight shapes post-loading: `at.check_pytree_equality()`

### 6. **Error Handling & Validation**
- Use `beartype` for runtime type checks (via `@at.typecheck`)
- Assertions for shape/dtype mismatches in training (helps catch data bugs early)
- Custom error messages reference "keypath" (JAX tree path notation)
- Logging via `logging.getLogger("openpi")` module-level

## Integration Points & External Dependencies

### 1. **LeRobot** (Data)
- `lerobot_dataset.LeRobotDataset`: Unified robot dataset interface
- Supports multiple robot platforms (ALOHA, DROID, XArm, etc.)
- Custom datasets via repack transforms

### 2. **RLDS** (DROID-specific)
- `droid_rlds_dataset.py`: Loads DROID dataset from TensorFlow RLDS format
- Handles action space conversion (continuous/discrete)
- Data filtering via `filter_dict_path` config

### 3. **Checkpoint Storage**
- Orbax for JAX checkpoints (distributed, efficient)
- SafeTensors for PyTorch weight interchange
- Local cache: `OPENPI_DATA_HOME` or `~/.cache/openpi`
- GCS buckets: `gs://openpi-assets/checkpoints/`

### 4. **Robot Platforms**
- **ALOHA**: ROS-based, 6-DoF dual-arm
- **DROID**: Franka arm with gripper
- **LIBERO**: Simulation benchmark
- Each has policy wrapper + environment interface

### 5. **W&B Integration**
- Optional experiment tracking (set `project_name` in config)
- Code logging via `wandb.run.log_code()`
- Resume runs via saved `wandb_id.txt` in checkpoint dir

## Testing & Debugging Tips

1. **Use fake data for quick iteration**:
   ```python
   obs, act = config.model.fake_obs(), config.model.fake_act()
   loss = model.compute_loss(rng, obs, act)
   ```

2. **Disable type checking for JAX tracing issues**:
   ```python
   with at.disable_typechecking():
       jitted_fn = jax.jit(fn)
   ```

3. **Check GPU memory** with `nvidia-smi` or `pynvml` (used in conftest.py)

4. **Data pipeline debugging**: Enable logging
   ```python
   logging.getLogger("openpi").setLevel(logging.DEBUG)
   ```

5. **Config overrides via CLI** are fully type-checked by tyro; use `--help` to explore

## When in Doubt

1. **Models**: Look at `model_test.py` for usage examples
2. **Training**: Check `examples/droid/README_train.md` for DROID-specific patterns
3. **Transforms**: Study `transforms_test.py` test cases
4. **Data loading**: Inspect `data_loader_test.py` for mock dataset patterns
5. **Checkpoints**: Use `download.maybe_download()` and inspect with `orbax.checkpoint.CheckpointManager`
