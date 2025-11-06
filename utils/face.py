from typing import List, Tuple
import os
import cv2
import numpy as np
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, weights_path: str, imgsz: int = 640, conf: float = 0.25, device: str = "cpu") -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ไม่พบไฟล์น้ำหนัก YOLOv8-face: {weights_path}")
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        # returns list of (x1, y1, x2, y2, conf)
        results = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        faces: List[Tuple[int, int, int, int, float]] = []
        if not results:
            return faces
        r = results[0]
        if r.boxes is None:
            return faces
        h, w = image_bgr.shape[:2]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, conf))
        return faces

    @staticmethod
    def crop_face(image_bgr: np.ndarray, box: Tuple[int, int, int, int], expand: float = 0.15) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * expand)
        dy = int(bh * expand)
        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy)
        nx2 = min(w, x2 + dx)
        ny2 = min(h, y2 + dy)
        return image_bgr[ny1:ny2, nx1:nx2]
