from PIL import Image
import io

def load_pil_from_bytes_or_path(x):
    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise ValueError("Unsupported input type for image.")
