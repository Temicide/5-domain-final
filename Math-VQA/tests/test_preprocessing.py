from pathlib import Path

from PIL import Image

from math_vqa.preprocessing import (
    load_rgb_image,
    preprocess_image,
    save_preprocessed_image,
    select_preprocess_name,
)


def make_image(path: Path, size: tuple[int, int] = (20, 10), mode: str = "L") -> Path:
    image = Image.new(mode, size, color=200)
    image.save(path)
    return path


def test_load_rgb_image_converts_to_rgb(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "gray.jpg", mode="L")

    image = load_rgb_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_upscale_preserves_aspect_ratio(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "small.jpg", size=(20, 10), mode="RGB")

    result = preprocess_image(image_path, "upscale")

    assert result.name == "upscale"
    assert result.image.size == (1024, 512)
    assert result.final_size == (1024, 512)


def test_select_preprocess_name_uses_specified_image_ids() -> None:
    assert select_preprocess_name("94") == "upscale"
    assert select_preprocess_name("156") == "contrast"
    assert select_preprocess_name("451") == "high_res"
    assert select_preprocess_name("7") == "raw"


def test_save_preprocessed_image_writes_rgb_jpeg(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "raw.jpg", size=(20, 10), mode="RGB")
    result = preprocess_image(image_path, "raw")

    saved = save_preprocessed_image(result, tmp_path / "prepared", image_id="7")

    assert saved.exists()
    assert saved.name == "7_raw.jpg"
    assert Image.open(saved).mode == "RGB"
