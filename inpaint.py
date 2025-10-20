import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imsave, imread
import os
import sys
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Image Inpainting using Laplace PDE and traditional filtering methods')
    parser.add_argument('--input', '-i', type=str, default='./img/0.jpeg',
                        help='Path to input image (default: ./img/0.jpeg)')
    parser.add_argument('--output', '-o', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--iterations', '-n', type=int, default=1000,
                        help='Number of iterations for Laplace PDE (default: 1000)')
    parser.add_argument('--brush-size', '-b', type=int, default=5,
                        help='Brush size for mask drawing (default: 5)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip displaying the final comparison plot')
    return parser.parse_args()

args = parse_args()

# Validate input image exists
if not os.path.exists(args.input):
    print(f"Error: Input image '{args.input}' not found!")
    sys.exit(1)

print(f"Loading image from: {args.input}")
image = img_as_float(imread(args.input)).copy()
h, w, _ = image.shape
print(f"Image dimensions: {w}x{h}")


mask = np.zeros((h, w), dtype=np.uint8)
drawing = False
ix, iy = -1, -1

def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mask, image_display
    brush_size = args.brush_size
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        image_display = (image * 255).astype(np.uint8).copy()
        overlay = np.zeros_like(image_display)
        overlay[mask > 0] = [0, 0, 255]
        image_display = cv2.addWeighted(image_display, 0.7, overlay, 0.3, 0)
        cv2.imshow(window_name, image_display)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        image_display = (image * 255).astype(np.uint8).copy()
        overlay = np.zeros_like(image_display)
        overlay[mask > 0] = [0, 0, 255]
        image_display = cv2.addWeighted(image_display, 0.7, overlay, 0.3, 0)
        cv2.imshow(window_name, image_display)

print("\n=== Interactive Mask Drawing ===")
print("Instructions:")
print("  - Hold LEFT MOUSE BUTTON and drag to draw the mask (shown in red)")
print("  - Press ESC when finished drawing to continue")
print("  - Press 'C' to clear the mask and start over")

window_name = 'Draw Mask (ESC to continue, C to clear)'
image_display = (image * 255).astype(np.uint8).copy()
image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_mask)
cv2.imshow(window_name, image_display)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c') or key == ord('C'):  # Clear mask
        mask = np.zeros((h, w), dtype=np.uint8)
        image_display = (image * 255).astype(np.uint8).copy()
        image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, image_display)
        print("Mask cleared!")
cv2.destroyAllWindows()

# Check if mask was drawn
if np.sum(mask) == 0:
    print("\nWarning: No mask was drawn! Please draw a mask to inpaint.")
    sys.exit(0)


print(f"\nMask area: {np.sum(mask > 0)} pixels ({100 * np.sum(mask > 0) / (h * w):.2f}% of image)")

mask_bool = (mask > 0)
damaged = image.copy()
damaged[mask_bool] = 0.0


def laplace_inpaint_channel(channel, mask, num_iters=1000):
    """
    Inpaint a single channel using Laplace equation-based PDE method.
    
    Args:
        channel: 2D array representing one color channel
        mask: Boolean mask indicating pixels to inpaint
        num_iters: Number of iterations for convergence
    
    Returns:
        Inpainted channel
    """
    u = channel.copy()
    for _ in tqdm(range(num_iters), desc="Laplace Inpainting", ncols=80):
        u_old = u.copy()
        u[mask] = 0.25 * (
            np.roll(u_old, 1, axis=0)[mask] +
            np.roll(u_old, -1, axis=0)[mask] +
            np.roll(u_old, 1, axis=1)[mask] +
            np.roll(u_old, -1, axis=1)[mask]
        )
    return u


print("\n=== Laplace PDE Inpainting ===")
inpainted_channels = []
for c in range(3):
    channel_name = ['Red', 'Green', 'Blue'][c]
    print(f"Processing {channel_name} channel...")
    inpainted = laplace_inpaint_channel(damaged[:, :, c], mask_bool, num_iters=args.iterations)
    inpainted_channels.append(inpainted)
inpainted_color = np.stack(inpainted_channels, axis=2)


psnr_values, ssim_values = [], []
for c in range(3):
    original_c = image[:, :, c][mask_bool]
    restored_c = inpainted_color[:, :, c][mask_bool]
    psnr = peak_signal_noise_ratio(original_c, restored_c, data_range=1.0)
    ssim = structural_similarity(original_c, restored_c, data_range=1.0)
    psnr_values.append(psnr)
    ssim_values.append(ssim)

print("\n=== Laplace PDE Results ===")
print(f"PSNR (R, G, B): {psnr_values[0]:.2f}, {psnr_values[1]:.2f}, {psnr_values[2]:.2f} dB")
print(f"SSIM (R, G, B): {ssim_values[0]:.4f}, {ssim_values[1]:.4f}, {ssim_values[2]:.4f}")
print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")


print("\n=== Traditional Filtering Methods ===")
damaged_uint8 = (np.clip(damaged, 0, 1) * 255).astype(np.uint8)
damaged_bgr = cv2.cvtColor(damaged_uint8, cv2.COLOR_RGB2BGR)

print("Applying Gaussian Filter...")
gaussian_blur = cv2.GaussianBlur(damaged_bgr, (9, 9), 0)
gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB) / 255.0

print("Applying Mean Filter...")
mean_blur = cv2.blur(damaged_bgr, (9, 9))
mean_blur_rgb = cv2.cvtColor(mean_blur, cv2.COLOR_BGR2RGB) / 255.0

def evaluate(method_img, name=""):
    """
    Evaluate inpainting quality using PSNR and SSIM metrics.
    
    Args:
        method_img: Inpainted image to evaluate
        name: Name of the method for display
    
    Returns:
        Tuple of (PSNR values, SSIM values) for each channel
    """
    psnrs, ssims = [], []
    for c in range(3):
        orig = image[:, :, c][mask_bool]
        restored = method_img[:, :, c][mask_bool]
        psnr = peak_signal_noise_ratio(orig, restored, data_range=1.0)
        ssim = structural_similarity(orig, restored, data_range=1.0)
        psnrs.append(psnr)
        ssims.append(ssim)
    print(f"\n{name}:")
    print(f"  PSNR (R, G, B): {psnrs[0]:.2f}, {psnrs[1]:.2f}, {psnrs[2]:.2f} dB")
    print(f"  SSIM (R, G, B): {ssims[0]:.4f}, {ssims[1]:.4f}, {ssims[2]:.4f}")
    print(f"  Average PSNR: {np.mean(psnrs):.2f} dB")
    print(f"  Average SSIM: {np.mean(ssims):.4f}")
    return psnrs, ssims

mean_psnr, mean_ssim = evaluate(mean_blur_rgb, "Mean Filter")
gauss_psnr, gauss_ssim = evaluate(gaussian_blur_rgb, "Gaussian Filter")


print(f"\n=== Saving Results to '{args.output}' ===")
os.makedirs(args.output, exist_ok=True)
imsave(f"{args.output}/original.png", (image * 255).astype(np.uint8))
imsave(f"{args.output}/damaged.png", (damaged * 255).astype(np.uint8))
imsave(f"{args.output}/inpainted_laplace.png", (inpainted_color * 255).astype(np.uint8))
imsave(f"{args.output}/mean_filter.png", (mean_blur_rgb * 255).astype(np.uint8))
imsave(f"{args.output}/gaussian_filter.png", (gaussian_blur_rgb * 255).astype(np.uint8))
imsave(f"{args.output}/mask.png", mask)
print(f"✓ Saved individual result images")

# Create enhanced visualization with metrics
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Image Inpainting Methods Comparison', fontsize=16, fontweight='bold')

# Original and Damaged
axs[0, 0].imshow(image)
axs[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
axs[0, 0].axis('off')

axs[0, 1].imshow(damaged)
axs[0, 1].set_title("Damaged Image", fontsize=12, fontweight='bold')
axs[0, 1].axis('off')

# Mask visualization
mask_display = np.zeros_like(image)
mask_display[mask_bool] = [1, 0, 0]  # Red
axs[0, 2].imshow(mask_display)
axs[0, 2].set_title("Mask (Red Area)", fontsize=12, fontweight='bold')
axs[0, 2].axis('off')

# Laplace PDE
axs[1, 0].imshow(inpainted_color)
axs[1, 0].set_title(f"Laplace PDE\nPSNR: {np.mean(psnr_values):.2f} dB | SSIM: {np.mean(ssim_values):.4f}", 
                     fontsize=11, fontweight='bold')
axs[1, 0].axis('off')

# Mean Filter
axs[1, 1].imshow(mean_blur_rgb)
axs[1, 1].set_title(f"Mean Filter\nPSNR: {np.mean(mean_psnr):.2f} dB | SSIM: {np.mean(mean_ssim):.4f}", 
                     fontsize=11, fontweight='bold')
axs[1, 1].axis('off')

# Gaussian Filter
axs[1, 2].imshow(gaussian_blur_rgb)
axs[1, 2].set_title(f"Gaussian Filter\nPSNR: {np.mean(gauss_psnr):.2f} dB | SSIM: {np.mean(gauss_ssim):.4f}", 
                     fontsize=11, fontweight='bold')
axs[1, 2].axis('off')

plt.tight_layout()
comparison_path = f"{args.output}/comparison_all_methods.png"
plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
print(f"✓ Saved comparison plot: {comparison_path}")

if not args.no_display:
    plt.show()
else:
    print("Skipping plot display (--no-display flag set)")
    plt.close()

print("\n=== Summary ===")
print(f"Best method by PSNR: ", end="")
methods = {'Laplace PDE': np.mean(psnr_values), 
           'Mean Filter': np.mean(mean_psnr), 
           'Gaussian Filter': np.mean(gauss_psnr)}
best_method = max(methods, key=methods.get)
print(f"{best_method} ({methods[best_method]:.2f} dB)")

print(f"Best method by SSIM: ", end="")
methods_ssim = {'Laplace PDE': np.mean(ssim_values), 
                'Mean Filter': np.mean(mean_ssim), 
                'Gaussian Filter': np.mean(gauss_ssim)}
best_method_ssim = max(methods_ssim, key=methods_ssim.get)
print(f"{best_method_ssim} ({methods_ssim[best_method_ssim]:.4f})")

print(f"\nAll results saved to: {os.path.abspath(args.output)}")
print("Done!")
