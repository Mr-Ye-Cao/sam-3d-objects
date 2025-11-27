# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3D Objects is a foundation model from Meta that reconstructs full 3D shape geometry, texture, and layout from a single image. It converts masked objects in images into 3D models with pose, shape, texture, and layout, handling challenging scenarios like small objects, occlusions, and unusual poses.

## Environment Setup

Requires Linux 64-bit with NVIDIA GPU (32GB+ VRAM). Uses CUDA 12.1.

```bash
# Create environment
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# Install dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

# Inference dependencies
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# Required patch
./patching/hydra
```

## Download Checkpoints

Requires HuggingFace authentication (request access at https://huggingface.co/facebook/sam-3d-objects):

```bash
pip install 'huggingface-hub[cli]<1.0'
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

## Running Inference

Quick test:
```bash
python demo.py
```

API usage:
```python
import sys
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
image = load_image("path/to/image.png")
mask = load_single_mask("path/to/mask_folder", index=0)
output = inference(image, mask, seed=42)
output["gs"].save_ply("output.ply")
```

See `notebook/demo_single_object.ipynb` and `notebook/demo_multi_object.ipynb` for more examples.

## Architecture

### Two-Stage Pipeline

1. **Stage 1 - Sparse Structure Generation**: Predicts 3D voxel occupancy and object pose from image+mask
2. **Stage 2 - Structured Latent Generation**: Generates detailed 3D structure conditioned on sparse structure

### Key Components

- **`sam3d_objects/pipeline/`**: Inference pipelines
  - `inference_pipeline.py`: Base pipeline class with model loading and inference logic
  - `inference_pipeline_pointmap.py`: Extended pipeline using depth estimation (MoGe) for pointmaps

- **`sam3d_objects/model/backbone/`**: Neural network architectures
  - `dit/`: DiT-based models and condition embedders (DINO, pointmap)
  - `tdfy_dit/`: Core transformer models for sparse structure and latent flow
  - `generator/`: Flow matching and shortcut models for generation

- **`sam3d_objects/model/backbone/tdfy_dit/`**:
  - `models/`: Sparse structure flow, latent flow, VAE encoder/decoder
  - `modules/sparse/`: Sparse tensor operations (attention, conv, transformer blocks)
  - `representations/`: Output representations (Gaussian splats, mesh via FlexiCubes)
  - `renderers/`: Gaussian splatting renderer

- **`notebook/inference.py`**: Public-facing inference API with helper functions for visualization and video rendering

### Configuration

Models are configured via Hydra YAML files. The main config is at `checkpoints/hf/pipeline.yaml`. The pipeline instantiates models using `hydra.utils.instantiate()`.

### Output Formats

- Gaussian splats (`.ply`) - primary 3D representation
- GLB meshes - exported via FlexiCubes with optional texture baking
- Pose information (rotation quaternion, translation, scale)

## Testing

```bash
pytest
```

## Key Dependencies

- PyTorch 2.5.1 with CUDA 12.1
- pytorch3d (for 3D transforms and rendering)
- kaolin (for visualization)
- gsplat (for Gaussian splatting)
- MoGe (depth estimation)
- xformers (efficient attention)
- spconv-cu121 (sparse convolutions)
