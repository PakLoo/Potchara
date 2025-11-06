import cv2
from typing import Tuple

COLORS = {
    0: (0, 200, 0),   # สวมถูกต้อง - เขียว
    1: (0, 165, 255), # สวมไม่ถูกต้อง - ส้ม
    2: (255, 0, 0),   # ไม่สวม - แดง
}


def draw_box_with_label(img, box: Tuple[int, int, int, int], label: str, color_key: int, conf: float = None):
    x1, y1, x2, y2 = box
    color = COLORS.get(color_key, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = label if conf is None else f"{label} {conf:.2f}"
    ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
