from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance


LOW_RES_IMAGE_IDS = {"94", "101", "134", "140", "162", "200"}
CONTRAST_IMAGE_IDS = {"156"}
HIGH_RES_IMAGE_IDS = {"451", "569"}
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    name: str
    final_size: tuple[int, int]


def load_rgb_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _resize_by_max_side(image: Image.Image, max_side: int, upscale: bool) -> Image.Image:
    width, height = image.size
    current_max = max(width, height)
    if current_max == 0:
        raise ValueError("image has zero-sized dimension")
    if current_max == max_side:
        return image.copy()
    if current_max > max_side or upscale:
        scale = max_side / current_max
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, RESAMPLE)
    return image.copy()


def preprocess_image(image_path: str | Path, variant: str = "raw") -> PreprocessResult:
    image = load_rgb_image(image_path)
    if variant == "raw":
        processed = image
    elif variant == "upscale":
        processed = _resize_by_max_side(image, max_side=1024, upscale=True)
    elif variant == "contrast":
        processed = ImageEnhance.Contrast(image).enhance(1.6)
    elif variant == "high_res":
        processed = _resize_by_max_side(image, max_side=1568, upscale=False)
    else:
        raise ValueError(f"unknown preprocessing variant: {variant}")
    return PreprocessResult(image=processed, name=variant, final_size=processed.size)


def select_preprocess_name(image_id: object) -> str:
    image_id_text = str(image_id)
    if image_id_text in LOW_RES_IMAGE_IDS:
        return "upscale"
    if image_id_text in CONTRAST_IMAGE_IDS:
        return "contrast"
    if image_id_text in HIGH_RES_IMAGE_IDS:
        return "high_res"
    return "raw"


def save_preprocessed_image(result: PreprocessResult, output_dir: str | Path, image_id: object) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_id}_{result.name}.jpg"
    result.image.convert("RGB").save(output_path, format="JPEG", quality=95)
    return output_path
