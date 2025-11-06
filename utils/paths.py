import os
from typing import Dict, List

CLASS_NAME_TO_ID: Dict[str, int] = {
    "with_mask": 0,                 # สวมถูกต้อง
    "without_mask": 1               # ไม่สวม
}

ID_TO_PRETTY: Dict[int, str] = {
    0: "with_mask",
    1: "without_mask"
}

PRETTY_TO_ID: Dict[str, int] = {v: k for k, v in ID_TO_PRETTY.items()}

VALID_EXTS: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS
