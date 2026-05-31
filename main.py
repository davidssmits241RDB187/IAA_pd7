import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.metrics import structural_similarity as ssim

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    is_gray = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)

    img = img.astype(np.float32) / 255.0

    return img, is_gray


def add_impulse_noise(image, amount=0.2):

    noisy = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    num_noisy = int(total_pixels * amount)
    coords_h = np.random.randint(0, h, num_noisy)
    coords_w = np.random.randint(0, w, num_noisy)

    if image.ndim == 2:
        noisy[coords_h, coords_w] = np.random.rand(num_noisy)
    else:
        noisy[coords_h, coords_w, 0] = np.random.rand(num_noisy)
        noisy[coords_h, coords_w, 1] = np.random.rand(num_noisy)
        noisy[coords_h, coords_w, 2] = np.random.rand(num_noisy)

    return noisy.astype(np.float32)


def remove_noise(image):

    blurred = cv2.GaussianBlur(image, (5, 5), 3)

    return blurred.astype(np.float32)


def gamma_correction(image, gamma_value=2.0):

    corrected = np.power(image, gamma_value)

    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def histogram_linear_transformation(image):
    g_min = np.min(image)
    g_max = np.max(image)

    if g_max <= g_min:
        return image.copy()

    k = 1.0 / (g_max - g_min)

    result = k * (image - g_min)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def mean_brightness(img, is_gray):

    if is_gray:
        return float(np.mean(img) * 255)

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)

    return float(np.mean(hsv[:, :, 2]))


def activate_appropriate_contrast_processor(img, is_gray):

    mean_v = mean_brightness(img, is_gray)

    if mean_v > 200:
        processed = gamma_correction(img)
        title = "Gamma Correction"
    else:
        processed = histogram_linear_transformation(img)
        title = "Histogram Linear Transformation"

    return processed, title


def calculate_rmse(original, compared):

    return np.sqrt(np.mean((original - compared) ** 2))


def calculate_ssim(original, compared, is_gray):
    original_uint8 = (original * 255).astype(np.uint8)
    compared_uint8 = (compared * 255).astype(np.uint8)

    if is_gray:
        score = ssim(original_uint8, compared_uint8, data_range=255)
    else:
        score = ssim(original_uint8, compared_uint8, channel_axis=2, data_range=255)

    return float(score)


def save_image_comparison(original, noisy, processed, out_path, is_gray, title=""):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("Original")
    axes[1].set_title("Noisy")
    axes[2].set_title("Processed")

    if is_gray:
        axes[0].imshow(original, cmap="gray")
        axes[1].imshow(noisy, cmap="gray")
        axes[2].imshow(processed, cmap="gray")

    else:
        axes[0].imshow(
            cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        )

        axes[1].imshow(cv2.cvtColor((noisy * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        axes[2].imshow(
            cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close()


def save_quality_report(
    rmse_noisy, rmse_processed, ssim_noisy, ssim_processed, out_path
):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis("off")

    text = (
        "IMAGE QUALITY MEASURES\n\n"
        "Original vs Noisy\n"
        f"RMSE : {rmse_noisy:.6f}\n"
        f"SSIM : {ssim_noisy:.6f}\n\n"
        "Original vs Processed\n"
        f"RMSE : {rmse_processed:.6f}\n"
        f"SSIM : {ssim_processed:.6f}\n"
    )

    ax.text(0.05, 0.95, text, fontsize=14, va="top", family="monospace")

    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close()


def process_single_image(path, idx):

    original, is_gray = load_image(path)

    noisy = add_impulse_noise(original, 0.2)

    denoised = remove_noise(noisy)

    processed, method_name = activate_appropriate_contrast_processor(denoised, is_gray)

    rmse_noisy = calculate_rmse(original, noisy)

    rmse_processed = calculate_rmse(original, processed)

    ssim_noisy = calculate_ssim(original, noisy, is_gray)

    ssim_processed = calculate_ssim(original, processed, is_gray)

    base = os.path.splitext(os.path.basename(path))[0]

    comparison_out = os.path.join(OUTPUT_DIR, f"{idx:02d}_{base}_comparison.png")

    quality_out = os.path.join(OUTPUT_DIR, f"{idx:02d}_{base}_quality.png")

    save_image_comparison(
        original, noisy, processed, comparison_out, is_gray, method_name
    )

    save_quality_report(
        rmse_noisy, rmse_processed, ssim_noisy, ssim_processed, quality_out
    )


def main():

    files = ["pibble.jpg", "car.jpg", "beach.jpg"]

    for idx, file in enumerate(files, start=1):
        if os.path.exists(file):
            process_single_image(file, idx)
        else:
            print(f"File not found: {file}")


if __name__ == "__main__":
    main()
