"""Run 3D reconstruction on segmented tiger."""

import sys
sys.path.append("notebook")

from inference import Inference, load_image, load_single_mask

# Paths
config_path = "checkpoints/hf/pipeline.yaml"
image_path = "tiger_segmentation/image.png"
mask_folder = "tiger_segmentation"
output_path = "tiger_3d.ply"

print("Loading SAM 3D Objects model...")
inference = Inference(config_path, compile=False)

print(f"Loading image: {image_path}")
image = load_image(image_path)
print(f"Image shape: {image.shape}")

print(f"Loading mask from: {mask_folder}/0.png")
mask = load_single_mask(mask_folder, index=0)
print(f"Mask shape: {mask.shape}")

print("Running 3D reconstruction...")
output = inference(image, mask, seed=42)

print(f"Saving 3D model to: {output_path}")
output["gs"].save_ply(output_path)

print(f"\nDone! Your 3D tiger reconstruction has been saved to: {output_path}")
print("\nOutput info:")
print(f"  - Rotation: {output['rotation']}")
print(f"  - Translation: {output['translation']}")
print(f"  - Scale: {output['scale']}")
