#!/usr/bin/env python3
"""
Demo script for image inpainting with pre-defined masks.
This script demonstrates the inpainting capabilities without requiring interactive GUI.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imsave, imread
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Demo: Image Inpainting with predefined masks')
    parser.add_argument('--input', '-i', type=str, default='./img/0.jpeg',
                        help='Path to input image (default: ./img/0.jpeg)')
    parser.add_argument('--output', '-o', type=str, default='./demo-results',
                        help='Output directory for results (default: ./demo-results)')
    parser.add_argument('--iterations', '-n', type=int, default=500,
                        help='Number of iterations for Laplace PDE (default: 500)')
    parser.add_argument('--mask-type', '-m', type=str, default='center',
                        choices=['center', 'horizontal', 'vertical', 'random'],
                        help='Type of mask to apply (default: center)')
    parser.add_argument('--mask-size', type=float, default=0.2,
                        help='Relative size of mask (0.0-1.0, default: 0.2)')
    return parser.parse_args()


def create_mask(shape, mask_type='center', size=0.2):
    """
    Create a predefined mask for demonstration.
    
    Args:
        shape: Tuple of (height, width)
        mask_type: Type of mask ('center', 'horizontal', 'vertical', 'random')
        size: Relative size of the mask (0.0 to 1.0)
    
    Returns:
        Boolean mask array
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    
    if mask_type == 'center':
        # Create a rectangular mask in the center
        mask_h = int(h * size)
        mask_w = int(w * size)
        start_h = (h - mask_h) // 2
        start_w = (w - mask_w) // 2
        mask[start_h:start_h+mask_h, start_w:start_w+mask_w] = True
        
    elif mask_type == 'horizontal':
        # Create a horizontal stripe
        mask_h = int(h * size)
        start_h = (h - mask_h) // 2
        mask[start_h:start_h+mask_h, :] = True
        
    elif mask_type == 'vertical':
        # Create a vertical stripe
        mask_w = int(w * size)
        start_w = (w - mask_w) // 2
        mask[:, start_w:start_w+mask_w] = True
        
    elif mask_type == 'random':
        # Create random scattered mask
        num_pixels = int(h * w * size)
        indices = np.random.choice(h * w, num_pixels, replace=False)
        mask.flat[indices] = True
    
    return mask


def laplace_inpaint_channel(channel, mask, num_iters=500):
    """
    Inpaint a single channel using Laplace equation-based PDE method.
    """
    u = channel.copy()
    for _ in tqdm(range(num_iters), desc="Inpainting", ncols=80):
        u_old = u.copy()
        u[mask] = 0.25 * (
            np.roll(u_old, 1, axis=0)[mask] +
            np.roll(u_old, -1, axis=0)[mask] +
            np.roll(u_old, 1, axis=1)[mask] +
            np.roll(u_old, -1, axis=1)[mask]
        )
    return u


def evaluate_inpainting(original, inpainted, mask):
    """
    Evaluate inpainting quality.
    """
    psnr_values = []
    ssim_values = []
    
    for c in range(3):
        orig = original[:, :, c][mask]
        restored = inpainted[:, :, c][mask]
        psnr = peak_signal_noise_ratio(orig, restored, data_range=1.0)
        ssim = structural_similarity(orig, restored, data_range=1.0)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    return psnr_values, ssim_values


def main():
    args = parse_args()
    
    # Load image
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found!")
        return
    
    print(f"Loading image: {args.input}")
    image = img_as_float(imread(args.input))
    h, w, _ = image.shape
    print(f"Image size: {w}x{h}")
    
    # Create mask
    print(f"Creating {args.mask_type} mask...")
    mask = create_mask((h, w), args.mask_type, args.mask_size)
    print(f"Mask area: {np.sum(mask)} pixels ({100 * np.sum(mask) / (h * w):.2f}%)")
    
    # Create damaged image
    damaged = image.copy()
    damaged[mask] = 0.0
    
    # Inpaint with Laplace PDE
    print(f"\nPerforming Laplace PDE inpainting ({args.iterations} iterations)...")
    inpainted_channels = []
    for c in range(3):
        channel_name = ['Red', 'Green', 'Blue'][c]
        print(f"Channel: {channel_name}")
        inpainted = laplace_inpaint_channel(damaged[:, :, c], mask, args.iterations)
        inpainted_channels.append(inpainted)
    inpainted_color = np.stack(inpainted_channels, axis=2)
    
    # Evaluate
    print("\nEvaluating results...")
    psnr_vals, ssim_vals = evaluate_inpainting(image, inpainted_color, mask)
    print(f"PSNR (R, G, B): {psnr_vals[0]:.2f}, {psnr_vals[1]:.2f}, {psnr_vals[2]:.2f} dB")
    print(f"SSIM (R, G, B): {ssim_vals[0]:.4f}, {ssim_vals[1]:.4f}, {ssim_vals[2]:.4f}")
    print(f"Average PSNR: {np.mean(psnr_vals):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_vals):.4f}")
    
    # Traditional methods
    print("\nComparing with traditional methods...")
    damaged_uint8 = (np.clip(damaged, 0, 1) * 255).astype(np.uint8)
    damaged_bgr = cv2.cvtColor(damaged_uint8, cv2.COLOR_RGB2BGR)
    
    mean_blur = cv2.blur(damaged_bgr, (9, 9))
    mean_blur_rgb = cv2.cvtColor(mean_blur, cv2.COLOR_BGR2RGB) / 255.0
    mean_psnr, mean_ssim = evaluate_inpainting(image, mean_blur_rgb, mask)
    
    gaussian_blur = cv2.GaussianBlur(damaged_bgr, (9, 9), 0)
    gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB) / 255.0
    gauss_psnr, gauss_ssim = evaluate_inpainting(image, gaussian_blur_rgb, mask)
    
    print(f"Mean Filter - Avg PSNR: {np.mean(mean_psnr):.2f} dB, Avg SSIM: {np.mean(mean_ssim):.4f}")
    print(f"Gaussian Filter - Avg PSNR: {np.mean(gauss_psnr):.2f} dB, Avg SSIM: {np.mean(gauss_ssim):.4f}")
    
    # Save results
    print(f"\nSaving results to '{args.output}'...")
    os.makedirs(args.output, exist_ok=True)
    
    imsave(f"{args.output}/original.png", (image * 255).astype(np.uint8))
    imsave(f"{args.output}/damaged.png", (damaged * 255).astype(np.uint8))
    imsave(f"{args.output}/inpainted.png", (inpainted_color * 255).astype(np.uint8))
    imsave(f"{args.output}/mask.png", (mask * 255).astype(np.uint8))
    
    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Image Inpainting Demo - {args.mask_type.capitalize()} Mask', fontsize=14, fontweight='bold')
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original", fontsize=11)
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(damaged)
    axs[0, 1].set_title("Damaged", fontsize=11)
    axs[0, 1].axis('off')
    
    mask_display = np.zeros_like(image)
    mask_display[mask] = [1, 0, 0]
    axs[0, 2].imshow(mask_display)
    axs[0, 2].set_title("Mask", fontsize=11)
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(inpainted_color)
    axs[1, 0].set_title(f"Laplace PDE\nPSNR: {np.mean(psnr_vals):.2f} dB", fontsize=11)
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(mean_blur_rgb)
    axs[1, 1].set_title(f"Mean Filter\nPSNR: {np.mean(mean_psnr):.2f} dB", fontsize=11)
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(gaussian_blur_rgb)
    axs[1, 2].set_title(f"Gaussian Filter\nPSNR: {np.mean(gauss_psnr):.2f} dB", fontsize=11)
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{args.output}/comparison.png", dpi=150, bbox_inches='tight')
    print(f"✓ Results saved to {os.path.abspath(args.output)}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
