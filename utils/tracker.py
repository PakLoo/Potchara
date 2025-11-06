# utils/tracker.py
import numpy as np
from collections import deque

__all__ = ["SimpleTracker"]

class _Track:
    """สถานะของวัตถุ 1 ตัวพร้อม EMA + hysteresis + majority window"""
    def __init__(self, tid: int, bbox, prob_vec, window_N=9, ema_beta=0.85):
        self.id = int(tid)
        self.bbox = [int(b) for b in bbox]            # [x1,y1,x2,y2]
        self.age = 0                                  # อายุ (เฟรมที่ไม่ได้แมตช์)
        self.hits = 1                                 # เฟรมที่แมตช์สำเร็จสะสม

        # smoothing
        self.window_N = int(window_N)
        self.window_preds = deque(maxlen=self.window_N)
        self.ema_beta = float(ema_beta)
        self.p_ema = prob_vec.astype(np.float32).copy()  # EMA ของ prob
        self.window_preds.append(int(np.argmax(prob_vec)))

        # hysteresis/debounce state
        self.stable_label = int(np.argmax(self.p_ema))
        self.switch_candidate = None
        self.switch_count = 0

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        ua = max(0, ax2-ax1) * max(0, ay2-ay1)
        ub = max(0, bx2-bx1) * max(0, by2-by1)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def update_ema(self, prob_vec, low_conf_filter=0.55):
        """อัปเดต EMA เฉพาะเฟรมที่มี max(prob) ≥ low_conf_filter เพื่อกัน noise"""
        m = float(np.max(prob_vec))
        if m < float(low_conf_filter):
            return
        self.p_ema = self.ema_beta * self.p_ema + (1.0 - self.ema_beta) * prob_vec

    def push_pred(self, prob_vec):
        self.window_preds.append(int(np.argmax(prob_vec)))

    def decide_label(self, t_on=0.7, t_off=0.6, margin=0.1, K=3):
        """
        คืน (final_label, final_conf) จาก EMA + hysteresis/debounce + majority window
        - ถ้า label ใหม่มี prob_ema สูงกว่า t_on และชนะเดิม ≥ margin ต่อเนื่อง K เฟรม -> สลับ
        - ถ้า prob_ema ของ label ปัจจุบัน < t_off -> fallback majority window
        """
        prob = self.p_ema
        curr = self.stable_label
        top = int(np.argmax(prob))
        top_conf = float(prob[top])
        cur_conf = float(prob[curr])

        ready_to_switch = (top != curr) and (top_conf >= t_on) and ((top_conf - cur_conf) >= margin)

        if ready_to_switch:
            if self.switch_candidate != top:
                self.switch_candidate = top
                self.switch_count = 1
            else:
                self.switch_count += 1

            if self.switch_count >= int(K):
                self.stable_label = top
                self.switch_candidate = None
                self.switch_count = 0
                curr = self.stable_label
                cur_conf = float(prob[curr])
        else:
            # ค่อย ๆ ลืม candidate ถ้ายังไม่ถึงเกณฑ์
            if top == self.switch_candidate:
                self.switch_count = max(0, self.switch_count - 1)
                if self.switch_count == 0:
                    self.switch_candidate = None
            else:
                self.switch_candidate = None
                self.switch_count = 0

        final_label = self.stable_label
        final_conf  = float(prob[final_label])

        if final_conf < t_off and len(self.window_preds) > 0:
            counts = np.bincount(self.window_preds, minlength=prob.shape[0])
            mw_label = int(np.argmax(counts))
            final_label = mw_label
            final_conf  = float(counts[mw_label] / max(1, int(np.sum(counts))))

        return final_label, final_conf


class SimpleTracker:
    """
    ByteTrack-style (ฉบับเบา): Greedy IoU matching + อายุ track + EMA + hysteresis
    ไม่ใช้ embedding, เหมาะกับโจทย์ใบหน้า/กล้องคงที่
    """
    def __init__(self, iou_threshold=0.3, max_age=30, ema_beta=0.85, window_N=9,
                 t_on=0.7, t_off=0.6, margin=0.1, K=3, low_conf_filter=0.55):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.ema_beta = float(ema_beta)
        self.window_N = int(window_N)
        self.t_on = float(t_on)
        self.t_off = float(t_off)
        self.margin = float(margin)
        self.K = int(K)
        self.low_conf_filter = float(low_conf_filter)

        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        ua = max(0, ax2-ax1) * max(0, ay2-ay1)
        ub = max(0, bx2-bx1) * max(0, by2-by1)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """
        detections: [{'box':[x1,y1,x2,y2], 'prob':np.array([p0,p1,...])}, ...]
        return: list[_Track]
        """
        used = set()
        new_tracks = []

        # 1) greedy match กับทุก track
        for tr in self.tracks:
            best_j, best_iou = None, 0.0
            for j, det in enumerate(detections):
                if j in used: 
                    continue
                iou_val = self._iou(tr.bbox, det['box'])
                if iou_val >= self.iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_j is not None:
                det = detections[best_j]
                used.add(best_j)
                tr.age = 0
                tr.hits += 1
                tr.bbox = [int(v) for v in det['box']]
                tr.update_ema(det['prob'], low_conf_filter=self.low_conf_filter)
                tr.push_pred(det['prob'])
                new_tracks.append(tr)
            else:
                tr.age += 1
                if tr.age < self.max_age:
                    new_tracks.append(tr)

        # 2) ที่เหลือสร้าง track ใหม่
        for j, det in enumerate(detections):
            if j in used:
                continue
            t = _Track(self.next_id, det['box'], det['prob'],
                       window_N=self.window_N, ema_beta=self.ema_beta)
            self.next_id += 1
            new_tracks.append(t)

        self.tracks = new_tracks
        return self.tracks

    # helper: แปลงผลเป็น (id, bbox, label, conf)
    def finalized(self, id_to_label_fn, **decide_kw):
        outs = []
        for tr in self.tracks:
            lid, conf = tr.decide_label(
                t_on     = decide_kw.get("t_on", self.t_on),
                t_off    = decide_kw.get("t_off", self.t_off),
                margin   = decide_kw.get("margin", self.margin),
                K        = decide_kw.get("K", self.K),
            )
            outs.append({
                "id": tr.id,
                "bbox": tr.bbox,
                "label_id": lid,
                "label": id_to_label_fn(lid),
                "conf": float(conf)
            })
        return outs
