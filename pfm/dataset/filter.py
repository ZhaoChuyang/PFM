from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MAX_PIXELS = 50_000_000


def check_image_filter(
    image: Image.Image,
    caption: str,
    min_resolution: int | None = None,
    min_aspect_ratio: float | None = None,
    max_aspect_ratio: float | None = None,
    **kwargs,
) -> bool:
    w, h = image.size
    if w * h > MAX_PIXELS:
        return False
    if min_resolution is not None and min(h, w) < min_resolution:
        return False
    aspect = w / h
    if min_aspect_ratio is not None and aspect < min_aspect_ratio:
        return False
    if max_aspect_ratio is not None and aspect > max_aspect_ratio:
        return False
    return True
