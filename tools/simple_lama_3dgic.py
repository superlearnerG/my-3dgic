import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SimpleLaMa on 3DGIC round RGB/depth reference views.")
    parser.add_argument("--inpaint_dir", required=True, help="Path to data_work/inpaint_2d_unseen_mask.")
    parser.add_argument("--stems", nargs="+", required=True, help="Selected reference view stems to inpaint.")
    parser.add_argument("--mode", choices=("color", "depth", "both"), default="both")
    parser.add_argument("--device", default="cuda", help="Device passed to simple_lama_inpainting.SimpleLama.")
    parser.add_argument("--mask_dilation", type=int, default=0)
    parser.add_argument("--mask_threshold", type=int, default=0)
    return parser.parse_args()


def init_lama(device):
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise ImportError(
            "simple_lama_inpainting is required in the current environment."
        ) from exc
    return SimpleLama(device=device)


def find_image_for_stem(folder, stem):
    for ext in IMAGE_EXTENSIONS:
        path = folder / f"{stem}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found for stem '{stem}' in {folder}")


def load_binary_mask(mask_path, threshold, dilation_radius):
    if dilation_radius < 0:
        raise ValueError(f"--mask_dilation must be >= 0, got {dilation_radius}")

    with Image.open(mask_path) as mask:
        mask_np = np.asarray(mask.convert("L"), dtype=np.uint8)
    binary = np.where(mask_np > threshold, 255, 0).astype(np.uint8)

    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    return Image.fromarray(binary, mode="L")


def crop_output_to_input_size(output, input_size, input_path):
    if output.size == input_size:
        return output

    input_width, input_height = input_size
    output_width, output_height = output.size
    if output_width < input_width or output_height < input_height:
        raise RuntimeError(
            f"SimpleLaMa output is smaller than input for {input_path}: "
            f"input={input_size}, output={output.size}"
        )
    return output.crop((0, 0, input_width, input_height))


def run_one(simple_lama, image_path, mask_path, threshold, dilation_radius):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    mask = load_binary_mask(mask_path, threshold, dilation_radius)
    if image.size != mask.size:
        raise RuntimeError(
            f"Image/mask size mismatch: {image_path} size={image.size}, "
            f"{mask_path} size={mask.size}"
        )

    output = simple_lama(image, mask).convert("RGB")
    output = crop_output_to_input_size(output, image.size, image_path)
    output.save(image_path)
    print(f"[SimpleLaMa][3DGIC] {image_path}")


def main():
    args = parse_args()
    inpaint_dir = Path(args.inpaint_dir).expanduser().resolve()
    color_dir = inpaint_dir / "images"
    depth_dir = inpaint_dir / "depth_removal"

    if not inpaint_dir.is_dir():
        raise FileNotFoundError(f"Inpaint directory not found: {inpaint_dir}")
    if args.mode in ("color", "both") and not color_dir.is_dir():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")
    if args.mode in ("depth", "both") and not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    simple_lama = init_lama(args.device)
    for stem in args.stems:
        mask_path = inpaint_dir / f"{stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if args.mode in ("color", "both"):
            run_one(
                simple_lama,
                find_image_for_stem(color_dir, stem),
                mask_path,
                args.mask_threshold,
                args.mask_dilation,
            )

        if args.mode in ("depth", "both"):
            depth_path = find_image_for_stem(depth_dir, stem)
            range_path = depth_dir / f"{stem}_range.pt"
            if not range_path.is_file():
                raise FileNotFoundError(f"Depth range file not found: {range_path}")
            run_one(
                simple_lama,
                depth_path,
                mask_path,
                args.mask_threshold,
                args.mask_dilation,
            )


if __name__ == "__main__":
    main()
