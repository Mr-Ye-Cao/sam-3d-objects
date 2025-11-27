"""Segment tiger using SAM3 text prompt and save results for 3D reconstruction."""

import os
import sys

# Add SAM3 to path
sys.path.insert(0, "/home/ye/ml-experiments/sam3/sam3")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Enable TF32 for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Paths
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
image_path = "/home/ye/ml-experiments/sam3/sam-3d-objects/tiger-wearing-sword.png"
output_dir = "/home/ye/ml-experiments/sam3/sam-3d-objects/tiger_segmentation"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("Loading SAM3 model...")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

with torch.autocast("cuda", dtype=torch.bfloat16):
    model = build_sam3_image_model(bpe_path=bpe_path)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image size: {width}x{height}")

    # Create processor with lower confidence threshold to catch more of the tiger
    processor = Sam3Processor(model, confidence_threshold=0.3)
    inference_state = processor.set_image(image)

    # Segment with text prompt "tiger"
    print("Segmenting with prompt: 'tiger'")
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="tiger")

    # Get results
    masks = inference_state.get("masks", [])
    boxes = inference_state.get("boxes", [])
    scores = inference_state.get("scores", [])

    print(f"Found {len(masks)} detection(s)")

    if len(masks) == 0:
        print("No tiger detected! Trying lower threshold...")
        processor = Sam3Processor(model, confidence_threshold=0.1)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt="tiger")
        masks = inference_state.get("masks", [])
        boxes = inference_state.get("boxes", [])
        scores = inference_state.get("scores", [])
        print(f"Found {len(masks)} detection(s) with lower threshold")

# Save results
image_np = np.array(image)

# Save original image
image.save(os.path.join(output_dir, "image.png"))
print(f"Saved: {output_dir}/image.png")

# Create visualization figure
fig, axes = plt.subplots(1, min(len(masks) + 1, 4), figsize=(16, 6))
if len(masks) == 0:
    axes = [axes]
elif len(masks) + 1 <= 1:
    axes = [axes]

# Show original image
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Process and save each mask
for i, (mask, score) in enumerate(zip(masks, scores)):
    if i >= 3:  # Limit to 3 masks
        break

    # Convert mask to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    # Ensure mask is 2D binary
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze()
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

    # Save mask
    mask_path = os.path.join(output_dir, f"{i}.png")
    Image.fromarray(mask_binary).save(mask_path)
    print(f"Saved mask {i}: {mask_path} (score: {score:.3f})")

    # Visualize
    overlay = image_np.copy()
    overlay[mask_np > 0.5] = overlay[mask_np > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5

    axes[i + 1].imshow(overlay.astype(np.uint8))
    axes[i + 1].set_title(f"Mask {i} (score: {score:.3f})")
    axes[i + 1].axis("off")

plt.tight_layout()
vis_path = os.path.join(output_dir, "segmentation_result.png")
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved visualization: {vis_path}")

print(f"\nResults saved to: {output_dir}/")
print("Files:")
print("  - image.png (original)")
for i in range(min(len(masks), 3)):
    print(f"  - {i}.png (mask)")
print("  - segmentation_result.png (visualization)")
