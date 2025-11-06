import threading
import time
import os
import sys
import queue
import csv
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np

# ========= Simple ByteTrack-Style Tracker with Temporal Smoothing =========
from collections import deque
import numpy as np

class Track:
    __slots__ = ("tid","box","age","hits","miss","p_ema","curr_label","switch_streak","window_preds","last_update_frame","alive","last_count_ts","last_save_ts","counted_label","counted_ts")
    def __init__(self, tid, box, now_f, num_classes=2, ema_beta=0.85, window_N=9):
        self.tid = tid
        self.box = box  # [x1,y1,x2,y2]
        self.age = 1
        self.hits = 1
        self.miss = 0
        self.p_ema = np.full((num_classes,), 1.0/num_classes, dtype=np.float32)
        self.curr_label = None
        self.switch_streak = 0
        self.window_preds = deque(maxlen=window_N)
        self.last_update_frame = now_f
        self.alive = True

        # =========== ใหม่: ตัวกันซ้ำตาม Track ===========
        self.last_count_ts = {"with_mask": 0.0, "without_mask": 0.0}
        self.last_save_ts  = 0.0  # ใช้กับรูป no-mask
        # นับแบบครั้งเดียวต่อ track-session
        self.counted_label = None  # "with_mask" | "without_mask" | None
        self.counted_ts = 0.0

class SimpleByteTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1,
                 ema_beta=0.85, t_on=0.7, t_off=0.6, margin=0.1, K=3, window_N=9,
                 num_classes=2):
        self.iou_thr = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.beta = float(ema_beta)
        self.t_on = float(t_on)
        self.t_off = float(t_off)
        self.margin = float(margin)
        self.K = int(K)
        self.window_N = int(window_N)
        self.num_classes = int(num_classes)

        self._next_id = 1
        self.tracks = []
        self._frame_idx = 0

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
        inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
        inter = iw * ih
        ua = max(0, ax2-ax1) * max(0, ay2-ay1)
        ub = max(0, bx2-bx1) * max(0, by2-by1)
        union = ua + ub - inter
        return inter/union if union>0 else 0.0

    def _greedy_match(self, dets):
        if not self.tracks or not dets:
            return [], list(range(len(self.tracks))), list(range(len(dets)))
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                iou = self._iou(t.box, d["box"])
                if iou >= self.iou_thr:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t, used_d = set(), set()
        matched = []
        for iou, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            matched.append((ti, di))
        un_t = [i for i in range(len(self.tracks)) if i not in used_t]
        un_d = [i for i in range(len(dets)) if i not in used_d]
        return matched, un_t, un_d

    def update(self, detections):
        self._frame_idx += 1
        for t in self.tracks:
            t.age += 1
            t.miss += 1

        matched, un_t, un_d = self._greedy_match(detections)

        for ti, di in matched:
            t = self.tracks[ti]
            t.box = detections[di]["box"]
            t.hits += 1
            t.miss = 0
            t.last_update_frame = self._frame_idx

        for ti in un_t:
            pass  # remain, will prune if too old

        for di in un_d:
            box = detections[di]["box"]
            nt = Track(self._next_id, box, self._frame_idx, num_classes=self.num_classes,
                       ema_beta=self.beta, window_N=self.window_N)
            self._next_id += 1
            self.tracks.append(nt)

        alive = []
        for t in self.tracks:
            if t.miss <= self.max_age:
                alive.append(t)
            else:
                t.alive = False
        self.tracks = alive

    def update_probs_and_get_label(self, track: Track, p_final: np.ndarray):
        # EMA only when confident frame
        if p_final.max() >= 0.55:
            track.p_ema = self.beta * track.p_ema + (1.0 - self.beta) * p_final
        new_idx = int(track.p_ema.argmax())
        old_idx = int(track.curr_label) if track.curr_label is not None else None

        track.window_preds.append(int(p_final.argmax()))

        def _majority_vote():
            if not track.window_preds:
                return new_idx
            vals, cnts = np.unique(np.array(track.window_preds), return_counts=True)
            return int(vals[int(cnts.argmax())])

        if old_idx is None:
            if track.hits >= 1:
                track.curr_label = _majority_vote()
            else:
                track.curr_label = new_idx
            return str(track.curr_label), float(track.p_ema[track.curr_label])

        if new_idx != old_idx:
            cond_on = (track.p_ema[new_idx] >= self.t_on) and \
                      (track.p_ema[new_idx] >= track.p_ema[old_idx] + self.margin)
            if cond_on:
                track.switch_streak += 1
                if track.switch_streak >= self.K:
                    track.curr_label = new_idx
                    track.switch_streak = 0
            else:
                track.switch_streak = 0
        else:
            if track.p_ema[old_idx] < self.t_off:
                track.curr_label = _majority_vote()
            track.switch_streak = 0

        return str(track.curr_label), float(track.p_ema[int(track.curr_label)])

    def outputs(self):
        return [t for t in self.tracks if t.hits >= self.min_hits and t.alive]


# ====== optional external libs ======
import pygame

# try optional imports (project utils)
try:
    from utils.paths import ID_TO_PRETTY
except Exception:
    ID_TO_PRETTY = None

try:
    # expected: class FaceDetector with method detect(frame_bgr) -> list of (x1,y1,x2,y2,conf)
    from utils.face import FaceDetector
except Exception:
    FaceDetector = None

try:
    from utils.viz import draw_box_with_label
except Exception:
    draw_box_with_label = None

# YOLOv8 fallback
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# torch for classifier
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

# ======================== [DB] MySQL Client ========================
# pip install mysql-connector-python
import mysql.connector
from mysql.connector import Error

class DBClient:
    def __init__(self,
                 host="127.0.0.1",
                 user="user_project",        # <<< ปรับตามของคุณ
                 password="1234",            # <<< ปรับตามของคุณ
                 database="project"):        # <<< ตาม project.sql
        self.cfg = dict(host=host, user=user, password=password, database=database)
        self.conn = None
        self.cur = None
        self.q = queue.Queue()
        self.running = False
        self.worker_th = None
        self._loc_cache = {}  # name -> id

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**self.cfg)
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            return True
        except Error as e:
            print("[DB] connect error:", e)
            return False

    def close(self):
        try:
            if self.cur: self.cur.close()
        except: pass
        try:
            if self.conn: self.conn.close()
        except: pass

    # ---------- async worker ----------
    def start_worker(self):
        if self.running: return
        if not self.conn or not self.conn.is_connected():
            self.connect()
        self.running = True
        self.worker_th = threading.Thread(target=self._loop, daemon=True)
        self.worker_th.start()

    def stop_worker(self):
        self.running = False
        self.q.put(("__STOP__", None))

    def _loop(self):
        while self.running:
            try:
                task, payload = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if task == "__STOP__":
                break
            try:
                if task == "ensure_location":
                    name = payload["name"]
                    cb = payload.get("callback")
                    loc_id = self._ensure_location_id(name)
                    if cb: cb(loc_id)
                elif task == "insert_no_mask":
                    self._insert_no_mask(**payload)
                elif task == "upsert_daily":
                    self._upsert_daily(**payload)
                elif task == "upsert_daily_delta":
                    self._upsert_daily_delta(**payload)
                elif task == "fetch_latest":
                    rows = self._fetch_latest(payload.get("limit", 100))
                    cb = payload.get("callback")
                    if cb: cb(rows)
            except Error as e:
                print("[DB] task error:", e)
                try: self.close()
                except: pass
                self.connect()
            except Exception as e:
                print("[DB] general error:", e)

    # ---------- public enqueue ----------
    def ensure_location_async(self, name, callback=None):
        self.q.put(("ensure_location", {"name": name, "callback": callback}))

    def upsert_daily_delta_async(self, day_str, location_name, with_mask_delta, without_mask_delta):
        self.q.put(("upsert_daily_delta", {
            "day_str": day_str,
            "location_name": location_name,
            "with_mask_delta": int(with_mask_delta),
            "without_mask_delta": int(without_mask_delta)
        }))

    def upsert_daily_async(self, day_str, location_name, with_mask, without_mask):
        self.q.put(("upsert_daily", {
            "day_str": day_str, "location_name": location_name,
            "with_mask": with_mask, "without_mask": without_mask
        }))

    def fetch_latest_async(self, limit, callback):
        self.q.put(("fetch_latest", {"limit": limit, "callback": callback}))

    # ---------- SQL ops ----------
    def _ensure_location_id(self, name: str) -> int:
        key = (name or "").strip()
        if not key:
            key = "ไม่ระบุ"
        if key in self._loc_cache:
            return self._loc_cache[key]
        # try select
        self.cur.execute("SELECT location_id FROM camera_location WHERE location_name=%s", (key,))
        row = self.cur.fetchone()
        if row:
            self._loc_cache[key] = int(row[0]); return self._loc_cache[key]
        # insert (unique uq_location_name)
        self.cur.execute("INSERT INTO camera_location (location_name) VALUES (%s)", (key,))
        self.cur.execute("SELECT location_id FROM camera_location WHERE location_name=%s", (key,))
        row = self.cur.fetchone()
        loc_id = int(row[0])
        self._loc_cache[key] = loc_id
        return loc_id

    def _insert_no_mask(self, capture_dt, image_name, location_name, score):
        loc_id = self._ensure_location_id(location_name)
        sql = """INSERT INTO no_mask_images (capture_datetime, image_name, location_id)
                 VALUES (%s, %s, %s)"""
        # NOTE: ตารางไม่มีคอลัมน์ score; เก็บเฉพาะชื่อไฟล์+เวลา+location_id ตามสคีมา
        self.cur.execute(sql, (capture_dt, image_name, loc_id))

    def insert_no_mask_async(self, capture_dt, image_name, location_name, score):
        self.q.put(("insert_no_mask", {
            "capture_dt": capture_dt,
            "image_name": image_name,
            "location_name": location_name,
            "score": float(score or 0.0),
        }))

    def _upsert_daily_delta(self, day_str, location_name, with_mask_delta, without_mask_delta):
        loc_id = self._ensure_location_id(location_name)
        stat_dt = day_str
        total_delta = int(with_mask_delta) + int(without_mask_delta)

        # ถ้าแถวใหม่ -> ใส่ค่า delta เป็นค่าเริ่มต้น
        # ถ้าแถวมีอยู่แล้ว -> บวกเพิ่มค่าเดิมด้วย VALUES(...)
        sql = """
        INSERT INTO daily_detection
            (stat_date, total_with_mask, total_without_mask, total_mask, wear_mask_percent, location_id)
        VALUES
            (%s, %s, %s, %s,
            CASE WHEN %s > 0 THEN ROUND(%s / %s * 100, 2) ELSE 0 END,
            %s)
        ON DUPLICATE KEY UPDATE
            total_with_mask     = total_with_mask     + VALUES(total_with_mask),
            total_without_mask  = total_without_mask  + VALUES(total_without_mask),
            total_mask          = total_mask          + VALUES(total_mask),
            wear_mask_percent   = CASE
                                    WHEN (total_mask + VALUES(total_mask)) > 0
                                    THEN ROUND(
                                        (total_with_mask + VALUES(total_with_mask)) /
                                        (total_mask + VALUES(total_mask)) * 100, 2
                                    )
                                    ELSE 0
                                END
        """
        # สำหรับค่าครั้งแรกต้องส่งตัวหาร/ตัวตั้งมาด้วย
        self.cur.execute(sql, (
            stat_dt,
            with_mask_delta,
            without_mask_delta,
            total_delta,
            total_delta,                # ตัวหารครั้งแรก
            with_mask_delta,            # ตัวตั้งครั้งแรก
            total_delta,                # ตัวหารครั้งแรก
            loc_id
        ))

    def _fetch_latest(self, limit=100):
        # รวมชื่อสถานที่ด้วย
        sql = """
        SELECT n.capture_datetime, n.image_name, c.location_name
        FROM no_mask_images n
        JOIN camera_location c ON c.location_id = n.location_id
        ORDER BY n.img_id DESC
        LIMIT %s
        """
        self.cur.execute(sql, (int(limit),))
        return self.cur.fetchall()
    
    # --- ดึงรายชื่อสถานที่ทั้งหมด (sync) ---
    def list_locations(self):
        try:
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return []
            cur = self.conn.cursor()
            cur.execute("SELECT location_name FROM camera_location ORDER BY location_name")
            rows = cur.fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            print("[DB] list_locations error:", e)
            return []

    # --- เพิ่มสถานที่ใหม่ (sync) และคืนชื่อที่เพิ่ม (หรือมีอยู่แล้ว) ---
    def add_location(self, name: str) -> bool:
        try:
            if not name.strip():
                return False
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return False
            # ใช้ UNIQUE(uq_location_name) ในตารางเพื่อกันซ้ำ
            cur = self.conn.cursor()
            cur.execute("INSERT IGNORE INTO camera_location (location_name) VALUES (%s)", (name.strip(),))
            self.conn.commit()
            return True
        except Exception as e:
            print("[DB] add_location error:", e)
            return False
        
    def fetch_daily_all(self):
        """คืน list ของแถวทั้งหมดในตาราง daily_detection รวมชื่อสถานที่"""
        try:
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return []
            cur = self.conn.cursor()
            cur.execute("""
                SELECT d.stat_date, d.total_with_mask, d.total_without_mask,
                    d.total_mask, d.wear_mask_percent, c.location_name
                FROM daily_detection d
                JOIN camera_location c ON c.location_id = d.location_id
                ORDER BY d.stat_date DESC
            """)
            return cur.fetchall()
        except Exception as e:
            print("[DB] fetch_daily_all error:", e)
            return []
        
    def fetch_daily_filtered(self, date_like: str = "", location_like: str = ""):
        try:
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return []

            q_date = (date_like or "").strip()
            q_loc  = (location_like or "").strip()

            where, params = [], []

            if q_date:
                # ดึงเฉพาะส่วนวันที่ (ซ้ายสุดก่อนช่องว่าง)
                q_date_only = q_date.split(" ")[0]  # เช่น "2025-10-14"
                where.append("DATE(d.stat_date) LIKE %s")
                params.append(f"{q_date_only}%")

            if q_loc:
                where.append("c.location_name LIKE %s")
                params.append(f"%{q_loc}%")

            where_sql = ("WHERE " + " AND ".join(where)) if where else ""
            sql = f"""
                SELECT d.stat_date, d.total_with_mask, d.total_without_mask,
                    d.total_mask, d.wear_mask_percent, c.location_name
                FROM daily_detection d
                JOIN camera_location c ON c.location_id = d.location_id
                {where_sql}
                ORDER BY d.stat_date DESC
            """
            cur = self.conn.cursor()
            cur.execute(sql, tuple(params))
            return cur.fetchall()
        except Exception as e:
            print("[DB] fetch_daily_filtered error:", e)
            return []
    
    def fetch_daily_by_date_location(self, day_str: str, location_name: str):
        """คืน daily_detection ของวันนั้นสำหรับสถานที่ที่กำหนด"""
        try:
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return []
            cur = self.conn.cursor()
            cur.execute(f"""
                SELECT d.stat_date, d.total_with_mask, d.total_without_mask,
                    d.total_mask, d.wear_mask_percent, c.location_name
                FROM daily_detection d
                JOIN camera_location c ON c.location_id = d.location_id
                WHERE DATE(d.stat_date) = %s
                AND c.location_name = %s
                ORDER BY d.stat_date DESC
            """, (day_str, location_name))
            return cur.fetchall()
        except Exception as e:
            print("[DB] fetch_daily_by_date_location error:", e)
            return []


    def fetch_no_mask_by_date_location(self, day_str: str, location_name: str):
        try:
            if not self.conn or not self.conn.is_connected():
                if not self.connect():
                    return []
            cur = self.conn.cursor()
            cur.execute(f"""
                SELECT n.capture_datetime, n.image_name, c.location_name
                FROM no_mask_images n
                JOIN camera_location c ON c.location_id = n.location_id
                WHERE DATE(n.capture_datetime) = %s
                AND c.location_name = %s
                ORDER BY n.capture_datetime DESC, n.img_id DESC
            """, (day_str, location_name))
            return cur.fetchall()
        except Exception as e:
            print("[DB] fetch_no_mask_by_date_location error:", e)
            return []






# ====================== [/DB] ======================


# ############## Canvas pill button ##############
class PillButton:
    def __init__(self, canvas: tk.Canvas, x: int, y: int, text: str, command,
                 w=110, h=46, radius=24,
                 fill="#FF2B2B", fill_hover="#D91F1F",
                 text_color="white", font=("TH Sarabun New", 22, "bold")):
        self.c = canvas
        self.x, self.y = x, y
        self.w, self.h, self.r = w, h, radius
        self.fill, self.fill_hover = fill, fill_hover
        self.text, self.command = text, command
        self.text_color, self.font = text_color, font
        self._hover = False
        self._build()

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.c.create_polygon(points, smooth=True, **kw)

    def _build(self):
        x1, y1 = self.x, self.y
        x2, y2 = x1 + self.w, y1 + self.h
        self.bg = self._rounded_rect(x1, y1, x2, y2, self.r, fill=self.fill, outline="#880000", width=2)
        self.label = self.c.create_text((x1+x2)//2, (y1+y2)//2,
                                        text=self.text, fill=self.text_color, font=self.font)
        for item in (self.bg, self.label):
            self.c.tag_bind(item, "<Enter>", self._on_enter)
            self.c.tag_bind(item, "<Leave>", self._on_leave)
            self.c.tag_bind(item, "<Button-1>", self._on_click)

    def _on_enter(self, _):
        if not self._hover:
            self._hover = True
            self.c.itemconfig(self.bg, fill=self.fill_hover)

    def _on_leave(self, _):
        if self._hover:
            self._hover = False
            self.c.itemconfig(self.bg, fill=self.fill)

    def _on_click(self, _):
        if callable(self.command):
            self.command()


# ############## Circle icon button ##############
class CircleIconButton:
    def __init__(self, canvas: tk.Canvas, x: int, y: int, r: int, text: str, command,
                 fill="#1DB954", fill_hover="#18A64B",
                 outline="#0B6A2A", text_color="white",
                 font=("TH Sarabun New", 18, "bold")):
        self.c = canvas
        self.r = r
        self.x, self.y = x, y
        self.text = text
        self.command = command
        self.fill, self.fill_hover = fill, fill_hover
        self.outline = outline
        self.text_color = text_color
        self.font = font
        self._hover = False
        self._build()

    def _build(self):
        x1, y1 = self.x - self.r, self.y - self.r
        x2, y2 = self.x + self.r, self.y + self.r
        self.bg_id = self.c.create_oval(x1, y1, x2, y2,
                                        fill=self.fill, outline=self.outline, width=2)
        self.txt_id = self.c.create_text(self.x, self.y, text=self.text,
                                         fill=self.text_color, font=self.font)
        for item in (self.bg_id, self.txt_id):
            self.c.tag_bind(item, "<Enter>", self._on_enter)
            self.c.tag_bind(item, "<Leave>", self._on_leave)
            self.c.tag_bind(item, "<Button-1>", self._on_click)

    def set_pos(self, x, y):
        self.x, self.y = x, y
        x1, y1 = x - self.r, y - self.r
        x2, y2 = x + self.r, y + self.r
        self.c.coords(self.bg_id, x1, y1, x2, y2)
        self.c.coords(self.txt_id, x, y)

    def _on_enter(self, _):
        if not self._hover:
            self._hover = True
            self.c.itemconfig(self.bg_id, fill=self.fill_hover)

    def _on_leave(self, _):
        if self._hover:
            self._hover = False
            self.c.itemconfig(self.bg_id, fill=self.fill)

    def _on_click(self, _):
        if callable(self.command):
            self.command()


class MaskDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Mask Detector - Webcam")
        # ---------------- window sizing ----------------
        master.geometry("900x600")
        master.minsize(700, 420)

        self.root_path = Path(__file__).parent

        # default model paths
        self.yolo_weights = str(self.root_path / "weights" / "yolov8n-face.pt")
        self.classifier_path = str(self.root_path / "models" / "image_baseline" / "mobilenetv2_best_2class.pt")

        self.camera_index = 0
        self.running = False
        self.show_boxes = True
        self.conf_thresh_fixed = 0.6  # ใช้กับ FaceDetector/YOLO
        self.thresh_var = tk.DoubleVar(value=self.conf_thresh_fixed)
        



        # ----- Tracking & Smoothing params -----
        self.iou_threshold = 0.3
        self.max_age = 30
        self.min_hits = 1
        self.ema_beta = 0.85
        self.t_on = 0.7
        self.t_off = 0.6
        self.margin = 0.1
        self.K = 3
        self.window_N = 9
        self.event_cooldown_sec = 3.0  # กันซ้ำนับต่อ track/label
        self.save_cooldown_sec  = 5.0  # กันสแปมเซฟรูปต่อ track
        self._tracker = None

        # model placeholders
        self.face_detector = None
        self.yolo_model = None
        self.classifier = None
        self.device = 'cpu'

        # frame queue (thread safe)
        self.frame_q = queue.Queue(maxsize=2)

        # save config
        self.save_dir = self.root_path / "บันทึกสถิติ"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / "records.csv"
        self.last_save_time = 0.0
        self.save_cooldown_seconds = 2.0

        # ---------- sound ----------
        self.alert_enabled = True
        self.alert_cooldown_sec = 10.0
        self._last_alert_time = 0.0
        self.alert_sound_path = self.root_path / "assets" / "alert.mp3"
        self.alert_sound = None
        try:
            pygame.mixer.init()
            if self.alert_sound_path.exists():
                self.alert_sound = pygame.mixer.Sound(str(self.alert_sound_path))
                print("✅ โหลดเสียงเตือนสำเร็จ:", self.alert_sound_path)
            else:
                print("⚠️ ไม่พบไฟล์เสียง:", self.alert_sound_path)
                self.alert_enabled = False
        except Exception as e:
            print("⚠️ ไม่สามารถเริ่ม pygame.mixer:", e)
            self.alert_sound = None
            self.alert_enabled = False

        # daily stats
        from collections import deque
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.with_mask_count_today = 0
        self.without_mask_count_today = 0
        self.recent_events = deque(maxlen=200)
        self.daily_csv_path = self.root_path / "บันทึกสถิติ" / "daily_stats.csv"
        self.show_running_totals = True

        # locations.csv (local combobox)
        self.locations_csv_path = self.save_dir / "locations.csv"

        self.location_name = None
        self.locations = self._load_locations()

        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---------------- [DB] init & start worker ----------------
        self.db = DBClient(
            host="127.0.0.1",
            user="user_project",   
            password="1234",       
            database="project"     
        )
        self.db.start_worker()
        

        # --- โหลดสถานที่จากฐานข้อมูลเป็นหลัก ---
        self.locations = self.db.list_locations()
        if not self.locations:
            # ถ้าเชื่อม DB ไม่ได้ หรือยังไม่มีข้อมูล ให้ fallback CSV เดิม
            self.locations = self._load_locations()


        # build UI
        self._build_ui()

        # start realtime pull (หลัง UI สร้างแล้ว)
        self.records_tree = None
        self._start_realtime_pull()

    # ---------- sound ----------
    def _play_alert(self):
        try:
            if self.alert_sound:
                self.alert_sound.play()
                print("⚠️ แจ้งเตือนผู้ไม่สวมหน้ากากอนามัย")
            else:
                self.master.bell()
        except Exception as e:
            print("⚠️ เล่นเสียงเตือนล้มเหลว:", e)

    def _init_tab_theme(self):
        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # --- เอา border/grip ของพื้นที่ client ออกทั้งหมด ---
        style.layout("Browser.TNotebook", [
            ("Notebook.client", {"sticky": "nswe"})  # ไม่มี border/frame อื่น ๆ
        ])

        # --- เอา focus ring (กรอบประ) ออกจาก Tab และทำให้ขอบเรียบ ---
        style.layout("Browser.TNotebook.Tab", [
            ("Notebook.tab", {
                "sticky": "nswe",
                "children": [  # ไม่มี Notebook.focus
                    ("Notebook.padding", {
                        "side": "top", "sticky": "nswe",
                        "children": [("Notebook.label", {"side": "top", "sticky": ""})]
                    })
                ]
            })
        ])

        # สี/ระยะของแท็บ (ไม่มีเส้นกรอบ)
        style.configure(
            "Browser.TNotebook",
            background="#F5F5F7",
            borderwidth=0,
            padding=0,
            tabmargins=[6, 2, 6, 0],
        )
        style.configure(
            "Browser.TNotebook.Tab",
            padding=[16, 8],
            background="#E9EAEE",
            foreground="#111",
            borderwidth=0,            # <- ไม่มีเส้นขอบแท็บ
            relief="flat"
        )
        style.map(
            "Browser.TNotebook.Tab",
            background=[("selected", "#FFFFFF"), ("active", "#F7F7FA")],
            foreground=[("selected", "#000")],
            # ไม่มี bordercolor เงื่อนไขใด ๆ = ไม่วาดเส้นขอบ
        )



    # ---------- UI scaffold ----------
    def _build_ui(self):
        self.page_container = ttk.Frame(self.master)
        self.page_container.pack(fill=tk.BOTH, expand=True)

        self.home_frame = self._build_home_page(self.page_container)
        self.live_frame = ttk.Frame(self.page_container)
        self._build_live_page(self.live_frame)

        self._show_page(self.home_frame)

    def _show_page(self, frame):
        for child in (self.home_frame, self.live_frame):
            child.pack_forget()
        frame.pack(fill=tk.BOTH, expand=True)

    # ---------- helpers draw ----------
    def _round_rect(self, canvas, x1, y1, x2, y2, r=20, **kwargs):
        r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
        pts = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return canvas.create_polygon(pts, smooth=True, **kwargs)

    def _draw_header(self, canvas, width):
        canvas.delete("all")
        TEAL = "#016E69"
        text = "ระบบตรวจจับการสวมหน้ากากอนามัย"
        h = 70
        r = 2
        x1, y1 = (width - 640)//2, 0
        x2, y2 = x1 + 640, h
        self._round_rect(canvas, x1, y1, x2, y2, r=r, fill=TEAL, outline="")
        canvas.create_text(width//2, h//2, text=text,
                           fill="white", font=("TH Sarabun New", 28, "bold"))

    def _build_home_page(self, container):
        BG_MAIN = "#C1C525"
        TEAL    = "#006D68"
        WHITE   = "#FFFFFF"

        self.master.configure(bg=BG_MAIN)

        parent = tk.Frame(container, bg=BG_MAIN)
        parent.pack(fill="both", expand=True)

        header_canvas = tk.Canvas(parent, bg=BG_MAIN, highlightthickness=0)
        header_canvas.pack(fill="x", pady=(40, 20))
        header_canvas.bind("<Configure>", lambda e: self._draw_header(header_canvas, e.width))

        card_frame = tk.Frame(parent, bg=TEAL)
        card_frame.place(relx=0.5, rely=0.55, anchor="center")

        PADX, PADY = 22, 18
        content_frame = tk.Frame(card_frame, bg=TEAL)
        content_frame.pack(padx=PADX, pady=PADY)

        tk.Label(content_frame, text="สถานที่ตั้งกล้อง",
                 font=("TH Sarabun New", 20, "bold"),
                 bg=TEAL, fg=WHITE).pack(anchor="center", pady=(4, 10))

        self.location_var = tk.StringVar()
        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure("Home.TCombobox", fieldbackground=WHITE, background=WHITE,
                        padding=6, arrowcolor="#333")

        self.location_cb = ttk.Combobox(content_frame, textvariable=self.location_var,
                                        values=self.locations, state="normal",
                                        width=30, style="Home.TCombobox")
        self.location_cb.pack(pady=(0, 18), ipady=6)
        if self.locations:
            self.locations = sorted(self.locations)
            self.location_cb["values"] = self.locations
            self.location_cb.set(self.locations[0])

        btn_canvas = tk.Canvas(content_frame, width=130, height=60, bg=TEAL, highlightthickness=0)
        btn_canvas.pack(pady=(2, 2))

        def draw_start_btn():
            btn_canvas.delete("all")
            w, h = 122, 48
            r = 20
            x1, y1 = 4, 4
            x2, y2 = x1 + w, y1 + h
            points = [
                x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
                x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
                x1, y2, x1, y2-r, x1, y1+r, x1, y1
            ]
            btn_canvas.create_polygon(points, smooth=True,
                                      fill="white", outline="black", width=2)
            btn_canvas.create_text((x1+x2)//2, (y1+y2)//2, text="เริ่ม",
                                   fill="black", font=("TH Sarabun New", 22, "bold"))

        draw_start_btn()

        def on_start_click(_=None):
            name = (self.location_var.get() or "").strip()
            if not name:
                messagebox.showwarning("กรอกสถานที่", "โปรดระบุชื่อสถานที่ก่อนเริ่ม")
                return
            # --- บันทึก DB เป็นหลัก ---
            ok = self.db.add_location(name)
            if not ok:
                # ถ้าเพิ่มใน DB ไม่สำเร็จ แจ้งเตือนและไม่ไปต่อ
                messagebox.showerror("เชื่อมต่อฐานข้อมูลไม่ได้",
                                    "ไม่สามารถเชื่อมต่อ/บันทึกสถานที่ลงฐานข้อมูลได้\nตรวจสอบการตั้งค่า DB แล้วลองใหม่")
                return

            # อัปเดตรายการใน combobox ให้มีชื่อใหม่นี้ด้วย (กันเคสเพิ่งเพิ่ม)
            vals = list(self.location_cb["values"])
            if name not in vals:
                vals.append(name)
                self.location_cb["values"] = sorted(vals)

            # (ไม่จำเป็นต้องเขียน CSV แล้ว แต่ถ้าจะเก็บไว้เป็น local ก็ได้)
            self._save_location_if_new(name)  # จะคงไว้ก็ได้ เป็น backup local

            self.location_name = name
            # [DB] ensure location exists in DB
            self.db.ensure_location_async(self.location_name)
            self._show_page(self.live_frame)
            self.master.after(50, self.start)

        btn_canvas.bind("<Button-1>", on_start_click)

        card_bg = tk.Canvas(card_frame, bg=TEAL, highlightthickness=0)
        card_bg.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Misc.lower(card_bg)

        def _redraw_card_bg(_=None):
            w = card_frame.winfo_width()
            h = card_frame.winfo_height()
            card_bg.delete("all")
            MARGIN = 2
            self._round_rect(card_bg, MARGIN, MARGIN, w - MARGIN, h - MARGIN,
                             r=18, fill=TEAL, outline="")

        card_frame.bind("<Configure>", _redraw_card_bg)
        parent.after(20, _redraw_card_bg)

        return parent

    def _build_live_page(self, parent):
        # โครงหลัก ซ้าย-ขวา
        root = ttk.Frame(parent); root.pack(fill=tk.BOTH, expand=True)

        SIDEBAR_W = 460  # ปรับตามชอบ 320-380 ได้
        root.columnconfigure(0, weight=1) # root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0) # root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        # ---------------- ซ้าย ----------------
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # แถบบน + Notebook (แยกกัน แต่อยู่ในเฟรมเดียวกัน)
        bar = tk.Frame(left, bg="#F5F5F7")
        bar.grid(row=0, column=0, sticky="nsew")
        bar.grid_columnconfigure(0, weight=1)   # ดันปุ่มไปชิดขวา
        bar.grid_rowconfigure(0, minsize=54)    # ความสูงหัวแถบ
        bar.grid_rowconfigure(1, weight=1)      # NOTEBOOK แถวล่างยืดเต็มสูง

        # กล่องปรับ Threshold (ขวาบน ใกล้ปุ่มหยุด)
        thr_box = tk.Frame(bar, bg="#F5F5F7")
        thr_box.grid(row=0, column=0, sticky="ne", padx=(0, 6), pady=(6, 0))

        tk.Label(thr_box, text="Threshold:", bg="#F5F5F7").pack(side="left", padx=(0, 4))

        def _apply_thresh_from_var(*_):
            try:
                v = float(self.thresh_var.get())
            except Exception:
                v = self.conf_thresh_fixed
            # บีบค่าให้อยู่ในช่วง 0.05-0.95 และปัด 2 ตำแหน่ง
            v = max(0.00, min(1.00, round(v, 2)))
            self.thresh_var.set(v)
            self.conf_thresh_fixed = v
            # อัปเดตตัวตรวจจับที่โหลดแบบ utils ถ้ามีแอตทริบิวต์ conf
            try:
                if self.face_detector is not None and hasattr(self.face_detector, "conf"):
                    self.face_detector.conf = float(v)
            except Exception:
                pass

        # Spinbox ปรับทีละ 0.05
        self.thresh_spin = tk.Spinbox(
            thr_box, from_=0.00, to=1.00, increment=0.05, format="%.2f",
            width=5, textvariable=self.thresh_var, command=_apply_thresh_from_var,
            justify="right"
        )
        self.thresh_spin.pack(side="left")

        # รองรับการพิมพ์/กด Enter หรือเลื่อนโฟกัส
        self.thresh_spin.bind("<Return>", _apply_thresh_from_var)
        self.thresh_spin.bind("<FocusOut>", _apply_thresh_from_var)


        # ปุ่มหยุด (ขวาสุดบนแถบบน)
        stop_host = tk.Canvas(bar, width=120, height=48, bg="#F5F5F7", highlightthickness=0)
        stop_host.grid(row=0, column=1, padx=(0, 6), pady=(6, 0), sticky="ne")
        def on_stop_click(): self._stop_with_confirm()
        self.stop_btn = PillButton(
            stop_host, x=6, y=4, text="หยุด", command=on_stop_click,
            w=96, h=40, radius=22,
            fill="#FF2B2B", fill_hover="#D91F1F",
            text_color="white", font=("TH Sarabun New", 20, "bold")
        )

        # Notebook (อยู่แถวล่างของ bar)
        self._init_tab_theme()
        self.tabs = ttk.Notebook(bar, style="Browser.TNotebook")
        self.tabs.configure(takefocus=False)
        self.tabs.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)

        # --- สร้างแท็บสองอัน: กล้อง/สรุป (ไม่มีแท็บหยุดแล้ว) ---
        tab_camera = ttk.Frame(self.tabs)
        tab_daily  = ttk.Frame(self.tabs)
        self.tabs.add(tab_camera, text="กล้องตรวจจับ", padding=0)
        self.tabs.add(tab_daily,  text="สรุปรายวัน", padding=0)

        # --------- เนื้อหาแท็บกล้อง ---------
        cam_wrap = ttk.Frame(tab_camera)
        cam_wrap.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(cam_wrap, bg="black", highlightthickness=0)
        self.canvas.pack(padx=6, pady=(6,6), fill=tk.BOTH, expand=True)
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor="center", image=None)

        # --------- เนื้อหาแท็บสรุปรายวัน ---------
        top_daily = ttk.Frame(tab_daily); top_daily.pack(fill="x", padx=8, pady=(8, 0))

        # ====== ภายใน tab_daily ======
        # คอนเทนเนอร์หลักของแท็บ (เอาไว้สลับ 2 โหมด)
        self.daily_stack = ttk.Frame(tab_daily)
        self.daily_stack.pack(fill=tk.BOTH, expand=True)

        # ---------- View 1: DAILY VIEW (ค้นหา + ตาราง daily_detection) ----------
        self.daily_view = ttk.Frame(self.daily_stack)
        self.daily_view.pack(fill=tk.BOTH, expand=True)

        # แถวค้นหา
        top_daily = ttk.Frame(self.daily_view); top_daily.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(top_daily, text="วันที่ (เช่น 2025-10 หรือ 2025-10-14)").pack(side="left")
        self.search_date_var = tk.StringVar()
        ttk.Entry(top_daily, textvariable=self.search_date_var, width=18).pack(side="left", padx=(6, 12))

        ttk.Label(top_daily, text="สถานที่").pack(side="left")
        self.search_place_var = tk.StringVar()
        ttk.Entry(top_daily, textvariable=self.search_place_var, width=18).pack(side="left", padx=(6, 12))

        ttk.Button(top_daily, text="ค้นหา", command=self._load_daily_table).pack(side="left", padx=(0, 6))
        ttk.Button(top_daily, text="รีเฟรช",
                command=lambda:[self.search_date_var.set(""), self.search_place_var.set(""), self._load_daily_table()]
                ).pack(side="left")

        # ตาราง daily
        daily_wrap = ttk.Frame(self.daily_view); daily_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.daily_tree = ttk.Treeview(daily_wrap, show="headings")
        vsb = ttk.Scrollbar(daily_wrap, orient="vertical", command=self.daily_tree.yview)
        hsb = ttk.Scrollbar(daily_wrap, orient="horizontal", command=self.daily_tree.xview)
        self.daily_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.daily_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        daily_wrap.rowconfigure(0, weight=1)
        daily_wrap.columnconfigure(0, weight=1)

        # ---------- View 2: IMAGES VIEW (เต็มหน้า) ----------
        self.images_view = ttk.Frame(self.daily_stack)  # เรายังไม่ pack จนกว่าจะสลับเข้า
        # แถบบน + ปุ่มย้อนกลับ
        img_top = ttk.Frame(self.images_view); img_top.pack(fill="x", padx=8, pady=(8, 0))
        self.images_title = ttk.Label(img_top, text="รูปภาพที่ไม่สวมหน้ากาก")
        self.images_title.pack(side="left")
        ttk.Button(img_top, text="ย้อนกลับ", command=lambda:self._show_daily_view()).pack(side="right")

        # ตารางรูป
        img_wrap = ttk.Frame(self.images_view); img_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.no_mask_full_tree = ttk.Treeview(img_wrap, show="headings")
        nm_vsb = ttk.Scrollbar(img_wrap, orient="vertical", command=self.no_mask_full_tree.yview)
        nm_hsb = ttk.Scrollbar(img_wrap, orient="horizontal", command=self.no_mask_full_tree.xview)
        self.no_mask_full_tree.configure(yscrollcommand=nm_vsb.set, xscrollcommand=nm_hsb.set)
        self.no_mask_full_tree.grid(row=0, column=0, sticky="nsew")
        nm_vsb.grid(row=0, column=1, sticky="ns")
        nm_hsb.grid(row=1, column=0, sticky="ew")
        img_wrap.rowconfigure(0, weight=1)
        img_wrap.columnconfigure(0, weight=1)

        # คอลัมน์ตารางรูป (จะโชว์วันเวลา + ชื่อไฟล์; ถ้าอยากซ่อนสถานที่จริง ๆ ไม่ต้องใส่คอลัมน์สถานที่)
        self.no_mask_full_cols = ["วันเวลา", "ไฟล์รูป", "สถานที่"]
        self.no_mask_full_tree["columns"] = self.no_mask_full_cols
        for h in self.no_mask_full_cols:
            w = 180 if h == "วันเวลา" else 300 if h == "ไฟล์รูป" else 220
            self.no_mask_full_tree.heading(h, text=h)
            self.no_mask_full_tree.column(h, width=w, anchor="w")

        


    


        # โหลดตารางอัตโนมัติเมื่อสลับมาที่แท็บ "สรุปรายวัน"
        def _on_tab_changed(_):
            if self.tabs.tab(self.tabs.select(), "text") == "สรุปรายวัน":
                self._load_daily_table()
        self.tabs.bind("<<NotebookTabChanged>>", _on_tab_changed)

        # ---------------- ขวา: A/B คงอยู่เสมอ ----------------
        sidebar = tk.Frame(root, bg="#F3F3F3", highlightthickness=0, width=SIDEBAR_W)
        sidebar.grid(row=0, column=1, sticky="ns", padx=0, pady=0)
        # ล็อกไม่ให้วิดเจ็ตด้านในดันความกว้าง/ความสูง
        #sidebar.grid_propagate(False)
        # ตั้งน้ำหนักภายใน sidebar ให้วาง A/B แบบแบ่งครึ่งได้
        sidebar.rowconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)

        panel_a = tk.Frame(sidebar, bg="#FFFFFF", bd=1, relief="solid")
        panel_a.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6,3))
                # ----- Panel A : ตาราง daily ของ "วันนี้" -----
        self.panel_a = panel_a
        # เคลียร์ภายใน (กันรันซ้ำ)
        for w in panel_a.winfo_children(): w.destroy()

        tk.Label(panel_a, text="สรุปรายวัน (วันนี้)", bg="#FFFFFF",
                 font=("TH Sarabun New", 16, "bold")).pack(pady=(8, 0))

        a_wrap = ttk.Frame(panel_a); a_wrap.pack(fill="both", expand=True, padx=6, pady=6)
        self.a_tree = ttk.Treeview(a_wrap, show="headings", height=8)
        a_vsb = ttk.Scrollbar(a_wrap, orient="vertical", command=self.a_tree.yview)
        self.a_tree.configure(yscrollcommand=a_vsb.set)
        self.a_tree.grid(row=0, column=0, sticky="nsew")
        a_vsb.grid(row=0, column=1, sticky="ns")
        a_wrap.rowconfigure(0, weight=1); a_wrap.columnconfigure(0, weight=1)

        self.a_cols = ["สวมหน้ากาก", "ไม่สวมหน้ากาก", "รวมทั้งหมด", "% การสวม"]
        self.a_tree["columns"] = self.a_cols
        for h in self.a_cols:
            w = 60 if h == "% การสวม" else 70
            self.a_tree.heading(h, text=h)
            self.a_tree.column(h, width=w, anchor="center")


        panel_b = tk.Frame(sidebar, bg="#FFFFFF", bd=1, relief="solid")
        panel_b.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3,6))
        # ----- Panel B : ดูภาพเมื่อดับเบิลคลิกชื่อไฟล์ -----
        self.panel_b = panel_b
        for w in panel_b.winfo_children(): w.destroy()
        self.panel_b_canvas = tk.Canvas(panel_b, bg="#FFFFFF", highlightthickness=0)
        self.panel_b_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.panel_b_img_id = self.panel_b_canvas.create_image(0, 0, anchor="center")
        self._panel_b_photo = None

        def _on_no_mask_dbl(_):
            sel = self.no_mask_full_tree.selection()
            if not sel: return
            vals = self.no_mask_full_tree.item(sel[0], "values")
            if not vals: return
            img_name = str(vals[1])
            # รูปถูกเซฟไว้ใน self.save_dir ด้วยชื่อไฟล์นี้
            self._show_image_in_panel_b(self.save_dir / img_name)

        self.no_mask_full_tree.bind("<Double-1>", _on_no_mask_dbl)


        def _on_b_resize(_):
            # วาดภาพใหม่ให้พอดีเมื่อ resize
            if getattr(self, "_panel_b_last_path", None):
                self._show_image_in_panel_b(self._panel_b_last_path)
        self.panel_b_canvas.bind("<Configure>", _on_b_resize)


        # เริ่มดึงข้อมูลสดให้ Panel A
        self.master.after(500, self._refresh_panel_a_realtime)

        # สถานะล่าง
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(parent, textvariable=self.status_var).pack(anchor=tk.W, padx=12, pady=(0, 8))

    def _show_daily_view(self):
        # ซ่อน images_view แล้วโชว์ daily_view
        try:
            self.images_view.pack_forget()
        except Exception:
            pass
        self.daily_view.pack(fill=tk.BOTH, expand=True)

    def _show_images_view(self, day_str: str, location_name: str):
        # อัปเดตหัวข้อ
        self.images_title.config(text=f"รูปภาพที่ไม่สวมหน้ากาก • {day_str} • {location_name}")
        # ซ่อน daily_view แล้วโชว์ images_view
        try:
            self.daily_view.pack_forget()
        except Exception:
            pass
        self.images_view.pack(fill=tk.BOTH, expand=True)
        # โหลดข้อมูลรูป
        self._load_no_mask_table_full(day_str, location_name)

    def _load_no_mask_table_full(self, day_str: str, location_name: str):
        try:
            rows = self.db.fetch_no_mask_by_date_location(day_str, location_name)
        except Exception as e:
            self.log(f"โหลดรูป no_mask (เต็มหน้า) ล้มเหลว: {e}")
            rows = []

        if not hasattr(self, "no_mask_full_tree") or not self.no_mask_full_tree.winfo_exists():
            return

        for i in self.no_mask_full_tree.get_children():
            self.no_mask_full_tree.delete(i)

        for dt, img, loc in rows:
            self.no_mask_full_tree.insert("", "end", values=[str(dt), str(img), str(loc)])


    def _show_image_in_panel_b(self, img_path: Path | str):
        try:
            p = Path(img_path)
            if not p.exists():
                # ไฟล์ถูกเก็บชื่อไว้เฉย ๆ ให้ลองดูในโฟลเดอร์บันทึกสถิติ
                p2 = self.save_dir / str(p.name)
                p = p2 if p2.exists() else p
            if not p.exists():
                self.log(f"ไม่พบรูป: {p}")
                return

            # โหลดและปรับขนาดพอดี panel B
            img = Image.open(p).convert("RGB")
            W = self.panel_b_canvas.winfo_width() or 10
            H = self.panel_b_canvas.winfo_height() or 10
            img.thumbnail((max(1, W-12), max(1, H-12)))
            self._panel_b_photo = ImageTk.PhotoImage(img)
            self.panel_b_canvas.itemconfig(self.panel_b_img_id, image=self._panel_b_photo)
            self.panel_b_canvas.coords(self.panel_b_img_id, W//2, H//2)
            self._panel_b_last_path = str(p)
        except Exception as e:
            self.log(f"แสดงภาพใน B ล้มเหลว: {e}")
        def _open_image_in_B_from_full(_):
            sel = self.no_mask_full_tree.selection()
            if not sel: return
            vals = self.no_mask_full_tree.item(sel[0], "values")
            if not vals: return
            img_name = str(vals[1])
            self._show_image_in_panel_b(self.save_dir / img_name)

        self.no_mask_full_tree.bind("<Double-1>", _open_image_in_B_from_full)

            
           

    


    def _refresh_panel_a_realtime(self):
        """สรุป 'วันนี้' ของสถานที่ปัจจุบัน (self.location_name) แล้วแสดง 4 คอลัมน์"""
        today = datetime.now().strftime("%Y-%m-%d")
        place = self.location_name or "ไม่ระบุ"

        try:
            rows = self.db.fetch_daily_by_date_location(today, place)
        except Exception:
            rows = []

        # รวม (ปกติควรได้แถวเดียวอยู่แล้ว แต่เผื่อไว้)
        sum_with = sum(int(r[1] or 0) for r in rows)
        sum_without = sum(int(r[2] or 0) for r in rows)
        total = sum_with + sum_without
        pct = (sum_with / total * 100.0) if total > 0 else 0.0

        if hasattr(self, "a_tree") and self.a_tree.winfo_exists():
            for i in self.a_tree.get_children():
                self.a_tree.delete(i)
            self.a_tree.insert("", "end", values=[str(sum_with), str(sum_without), str(total), f"{pct:.2f}"])

        self.master.after(1000, self._refresh_panel_a_realtime)






    def _load_daily_table(self):
        """โหลดข้อมูล daily_detection ทั้งหมดจาก DB ใส่ self.daily_tree (รองรับค้นหา)"""
        try:
            date_q = self.search_date_var.get().strip() if hasattr(self, "search_date_var") else ""
            place_q = self.search_place_var.get().strip() if hasattr(self, "search_place_var") else ""

            if date_q or place_q:
                rows = self.db.fetch_daily_filtered(date_like=date_q, location_like=place_q)
            else:
                rows = self.db.fetch_daily_all()
        except Exception as e:
            self.log(f"โหลดสรุปรายวันล้มเหลว: {e}")
            rows = []

        if not hasattr(self, "daily_tree") or self.daily_tree is None:
            return

        cols = ["วันที่", "จำนวนสวมหน้ากากอนามัยต่อวัน", "จำนวนไม่สวมหน้ากากอนามัยต่อวัน", "ผลรวมสวมหน้ากากอนามัยต่อวัน", "เปอร์เซ็นต์สวมหน้ากากอนามัยต่อวัน", "สถานที่"]
        if list(self.daily_tree["columns"]) != cols:
            self.daily_tree["columns"] = cols
            for h in cols:
                w = 160 if h == "วันที่" else 120 if h in ("จำนวนสวมหน้ากากอนามัยต่อวัน","จำนวนไม่สวมหน้ากากอนามัยต่อวัน","ผลรวมสวมหน้ากากอนามัยต่อวัน") else 100 if h == "เปอร์เซ็นต์สวมหน้ากากอนามัยต่อวัน" else 200
                self.daily_tree.heading(h, text=h)
                self.daily_tree.column(h, width=w, anchor="center" if h in ("วันที่","จำนวนสวมหน้ากากอนามัยต่อวัน","จำนวนไม่สวมหน้ากากอนามัยต่อวัน","ผลรวมสวมหน้ากากอนามัยต่อวัน","เปอร์เซ็นต์สวมหน้ากากอนามัยต่อวัน",) else "w")

        for i in self.daily_tree.get_children():
            self.daily_tree.delete(i)
        for stat_dt, m_yes, m_no, total, pct, loc in rows:
            self.daily_tree.insert("", "end", values=[
                str(stat_dt), str(m_yes), str(m_no), str(total), f"{pct:.2f}", str(loc)
            ])
        def _on_daily_open_full(_):
            sel = self.daily_tree.selection()
            if not sel: return
            vals = self.daily_tree.item(sel[0], "values")
            if not vals: return
            # โครงคอลัมน์: ["วันที่-เวลา"(หรือ"วันที่"), "ใส่", "ไม่ใส่", "รวม", "% ใส่", "สถานที่"]
            day_str = str(vals[0]).split(" ")[0]
            loc = str(vals[5])
            self._show_images_view(day_str, loc)

        # ใช้ Double-Click เพื่อ “เปิดโหมดเต็มหน้า”
        if not hasattr(self, "_bind_daily_open_full"):
            self.daily_tree.bind("<Double-1>", _on_daily_open_full)
            self._bind_daily_open_full = True


    def _load_no_mask_table(self, day_str: str, location_name: str):
        try:
            rows = self.db.fetch_no_mask_by_date_location(day_str, location_name)
        except Exception as e:
            self.log(f"โหลดรูป no_mask ล้มเหลว: {e}")
            rows = []

        if not hasattr(self, "no_mask_full_tree") or not self.no_mask_full_tree.winfo_exists():
            return

        for i in self.no_mask_full_tree.get_children():
            self.no_mask_full_tree.delete(i)
        for dt, img, loc in rows:
            self.no_mask_full_tree.insert("", "end", values=[str(dt), str(img), str(loc)])

                # --- bind เมื่อเลือกแถวบน เพื่อโหลด no_mask ของวัน+สถานที่นั้น ---
        def _on_daily_select(_):
            sel = self.daily_tree.selection()
            if not sel: return
            vals = self.daily_tree.item(sel[0], "values")
            if not vals: return
            # คอลัมน์: [วันที่-เวลา/หรือวันที่, ใส่, ไม่ใส่, รวม, %, สถานที่]
            day_str = str(vals[0]).split(" ")[0]
            loc = str(vals[5])
            self._load_no_mask_table(day_str, loc)

        if not hasattr(self, "_bind_daily_once"):
            self.daily_tree.bind("<<TreeviewSelect>>", _on_daily_select)
            self._bind_daily_once = True






    # ---------- confirm ----------
    def _confirm(self, title: str, message: str) -> bool:
        win = tk.Toplevel(self.master)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        bg = "#6B6B6B"

        C = tk.Canvas(win, width=360, height=180, bg=bg, highlightthickness=0)
        C.pack()

        def rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
            pts = [
                x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
                x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
                x1, y2, x1, y2-r, x1, y1+r, x1, y1
            ]
            return canvas.create_polygon(pts, smooth=True, **kw)

        rounded_rect(C, 6, 6, 354, 174, 18, fill=bg, outline=bg)
        C.create_text(180, 40, text=title, fill="white", font=("TH Sarabun New", 22, "bold"))
        C.create_text(180, 78, text=message, fill="white", font=("TH Sarabun New", 16))

        result = {"ok": False}
        def do_ok(): result["ok"]=True; win.destroy()
        def do_cancel(): result["ok"]=False; win.destroy()

        PillButton(C, x=58, y=104, text="ตกลง", command=do_ok,
                   w=110, h=46, radius=22,
                   fill="#FFFFFF", fill_hover="#EAEAEA",
                   text_color="#000000", font=("TH Sarabun New", 18, "bold"))
        PillButton(C, x=200, y=104, text="ยกเลิก", command=do_cancel,
                   w=110, h=46, radius=22,
                   fill="#FFFFFF", fill_hover="#EAEAEA",
                   text_color="#000000", font=("TH Sarabun New", 18, "bold"))

        win.update_idletasks()
        W, H = 360, 180
        x = self.master.winfo_rootx() + (self.master.winfo_width() - W)//2
        y = self.master.winfo_rooty() + (self.master.winfo_height() - H)//2
        win.geometry(f"{W}x{H}+{x}+{y}")

        self.master.wait_window(win)
        return result["ok"]

    def _stop_with_confirm(self):
        if not self.running:
            self._show_page(self.home_frame)
            return
        if self._confirm("ยืนยันหยุดการตรวจจับ", "ต้องการหยุดกล้องและกลับหน้าแรกหรือไม่?"):
            self.stop()
            self.master.after(150, lambda: self._show_page(self.home_frame))

    # ---------- stats viewer ----------
    def _open_stats_viewer(self, default_tab="daily"):
        win = tk.Toplevel(self.master)
        win.title("สถิติการตรวจจับ")
        win.geometry("820x560+100+80")
        win.transient(self.master)

        # FIX: เมื่อหน้าต่างถูกปิด/ทำลาย ให้ล้าง self.records_tree กัน callback ทำงานใส่ widget ที่หายไป
        def _on_stats_window_close():
            self.records_tree = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_stats_window_close)
        win.bind("<Destroy>", lambda e: setattr(self, "records_tree", None))  # FIX

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)

        daily_tab = ttk.Frame(nb)
        rec_tab   = ttk.Frame(nb)
        nb.add(daily_tab, text="สรุปรายวัน (daily_stats.csv)")
        nb.add(rec_tab,   text="บันทึกภาพ (Realtime, DB)")

        if default_tab == "records":
            nb.select(rec_tab)

        # ---- daily tab: ยังอ่านจาก CSV ได้ตามเดิม ----
        def build_daily_tab(tab):
            top = ttk.Frame(tab); top.pack(fill="x", padx=10, pady=8)
            ttk.Label(top, text="วันที่ (เช่น 2025-10-09 หรือ 2025-10)").grid(row=0, column=0, sticky="w")
            date_var = tk.StringVar()
            ttk.Entry(top, textvariable=date_var, width=22).grid(row=0, column=1, padx=(6, 18))
            ttk.Label(top, text="สถานที่").grid(row=0, column=2, sticky="w")
            loc_var = tk.StringVar()
            ttk.Entry(top, textvariable=loc_var, width=22).grid(row=0, column=3, padx=(6, 18))
            btn = ttk.Button(top, text="ค้นหา / โหลด",
                             command=lambda: load_csv(self.daily_csv_path, tree, date_var.get(), loc_var.get(), "วันที่", "สถานที่"))
            btn.grid(row=0, column=4, padx=4)

            tree_wrap = ttk.Frame(tab); tree_wrap.pack(fill="both", expand=True, padx=10, pady=(0,10))
            tree = ttk.Treeview(tree_wrap, show="headings")
            vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree_wrap, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            tree_wrap.rowconfigure(0, weight=1)
            tree_wrap.columnconfigure(0, weight=1)

            load_csv(self.daily_csv_path, tree, "", "", "วันที่", "สถานที่")
            return tree

        def load_csv(path: Path, tree: ttk.Treeview, date_q: str, place_q: str, date_col: str, place_col: str):
            for c in tree.get_children():
                tree.delete(c)
            if not path.exists():
                self.log(f"ไม่พบไฟล์: {path}")
                return
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return
            header = rows[0]
            data = rows[1:]
            tree["columns"] = header
            for h in header:
                tree.heading(h, text=h)
                tree.column(h, width=max(80, len(h)*12), anchor="w")
            def norm(s): return (s or "").replace("\ufeff","").strip()
            date_idx = -1
            place_idx = -1
            for i, h in enumerate(header):
                nh = norm(h)
                if nh == norm(date_col): date_idx = i
                if nh == norm(place_col): place_idx = i
            dq = (date_q or "").strip()
            pq = (place_q or "").strip().lower()
            for row in data:
                if not row: continue
                ok = True
                if dq and date_idx >= 0:
                    if not row[date_idx].startswith(dq):
                        ok = False
                if ok and pq and place_idx >= 0:
                    if pq not in (row[place_idx] or "").lower():
                        ok = False
                if ok:
                    tree.insert("", "end", values=row)

        daily_tree = build_daily_tab(daily_tab)

        # ---- records tab (Realtime จาก DB) ----
        def build_records_tab(tab):
            top = ttk.Frame(tab); top.pack(fill="x", padx=10, pady=8)
            ttk.Label(top, text="รายชื่อล่าสุดจากฐานข้อมูล (อัปเดตทุก 1 วินาที)").pack(anchor="w")
            tree_wrap = ttk.Frame(tab); tree_wrap.pack(fill="both", expand=True, padx=10, pady=(6,10))
            tree = ttk.Treeview(tree_wrap, show="headings")
            vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree_wrap, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            tree_wrap.rowconfigure(0, weight=1)
            tree_wrap.columnconfigure(0, weight=1)
            return tree

        self.records_tree = build_records_tab(rec_tab)  # เก็บ ref สำหรับอัปเดตสด

    # ---------- realtime DB pull ----------
    def _start_realtime_pull(self):
        def tick():
            try:
                self.db.fetch_latest_async(limit=100, callback=self._on_latest_rows)
            except Exception as e:
                print("[UI] realtime pull err:", e)
            finally:
                self.master.after(1000, tick)
        self.master.after(1000, tick)

    def _on_latest_rows(self, rows):
        def update():
            tree = getattr(self, "records_tree", None)
            if not tree:
                return
            # FIX: หาก widget ถูกทำลายไปแล้ว ไม่ต้องอัปเดต
            try:
                if not tree.winfo_exists():
                    self.records_tree = None
                    return
            except tk.TclError:
                self.records_tree = None
                return

            try:
                cols = ["วันเวลา", "รูปภาพ", "สถานที่"]
                # ใช้ cget เพื่อหลีกเลี่ยงเคสที่ tree["columns"] โยน TclError
                current_cols = list(tree.cget("columns")) if tree.cget("columns") else []
                if current_cols != cols:
                    tree["columns"] = cols
                    for h in cols:
                        width = 180 if h == "วันเวลา" else 260 if h == "รูปภาพ" else 200
                        tree.heading(h, text=h)
                        tree.column(h, width=width, anchor="w")

                # เคลียร์และเติมใหม่
                for c in tree.get_children():
                    tree.delete(c)
                for (dtstr, img_name, loc_name) in rows:
                    tree.insert("", "end", values=[str(dtstr), str(img_name), str(loc_name)])
            except tk.TclError:
                # ถ้า widget ถูกทำลายระหว่างอัปเดต
                self.records_tree = None
                return
        self.master.after(0, update)

    # ---------- locations helpers ----------
    def _load_locations(self):
        import csv
        path = self.locations_csv_path
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(["location_ID", "location_name"])
            return []
        names, seen = [], set()
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = (r.get("location_name") or r.get("\ufefflocation_name") or "").strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower()); names.append(name)
        return names

    def _save_location_if_new(self, name: str):
        import csv
        path = self.locations_csv_path
        name = (name or "").strip()
        if not name: return
        if any(name.lower() == n.lower() for n in self.locations):
            return
        next_id = 1
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                ids = []
                for r in reader:
                    try:
                        ids.append(int((r.get("location_ID") or r.get("\ufefflocation_ID") or "0").strip() or "0"))
                    except: pass
                if ids: next_id = max(ids) + 1
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["location_ID", "location_name"])
            w.writerow([next_id, name])
        self.locations.append(name)
        if hasattr(self, "location_cb"):
            vals = list(self.location_cb["values"])
            if name not in vals:
                vals.append(name)
                self.location_cb["values"] = vals

    def log(self, text):
        self.status_var.set(text)
        print("[MaskDetector]", text)

    # ---------- run/stop ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self.log("Starting...")
        t = threading.Thread(target=self._setup_and_run, daemon=True)
        t.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.log("Stopping...")

    # ---------- save record (CSV + image + [DB]) ----------
    def _save_record(self, image_rgb, face_box, label, score):
        now = time.time()
        if now - self.last_save_time < self.save_cooldown_seconds:
            return
        self.last_save_time = now

        x1, y1, x2, y2 = face_box
        h, w = image_rgb.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image_rgb

        ts_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name = f"without_mask_{ts_file}.jpg"
        img_path = self.save_dir / img_name
        cv2.imwrite(str(img_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_header = (not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["วันเวลา", "รูปภาพ", "สถานที่"])
            place = self.location_name or "ไม่ระบุ"
            writer.writerow([ts_human, img_name, place])

        # ---------- [DB] insert no_mask_images ----------
        try:
            place = self.location_name or "ไม่ระบุ"
            self.db.insert_no_mask_async(
                capture_dt=ts_human,
                image_name=str(img_name),   # เก็บเฉพาะชื่อไฟล์ตาม schema
                location_name=place,
                score=float(score or 0.0)
            )
        except Exception as e:
            self.log(f"DB insert event error: {e}")

        self.log(f"Saved record.csv & image → {img_name}")

    # ---------- models ----------
    def _setup_models(self):
        if torch is not None and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # FaceDetector (utils)
        if FaceDetector is not None:
            try:
                self.face_detector = FaceDetector(
                    weights_path=str(self.yolo_weights),
                    conf=self.conf_thresh_fixed,
                    device=self.device
                )
                self.log("Loaded utils.FaceDetector")
            except Exception as e:
                self.log(f"utils.FaceDetector failed: {e}")
                self.face_detector = None

        # YOLO fallback
        if self.face_detector is None and YOLO is not None:
            try:
                self.yolo_model = YOLO(self.yolo_weights)
                self.log("Loaded YOLOv8 face model")
            except Exception as e:
                self.log(f"Failed to load YOLO model: {e}")
                self.yolo_model = None

        # classifier
        if torch is not None:
            try:
                ckpt = torch.load(self.classifier_path, map_location=self.device)
                from torchvision import models
                model = models.mobilenet_v2(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(in_features, 2)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    model_state = ckpt['model']
                else:
                    model_state = ckpt
                model.load_state_dict(model_state, strict=False)
                model.eval()
                model.to(self.device)
                self.classifier = model
                self.log("Loaded MobileNetV2 classifier")
            except Exception as e:
                self.classifier = None
                self.log(f"Failed to load classifier: {e}")
        else:
            self.log("torch not available - classifier disabled")

        if not Path(self.yolo_weights).exists():
            self.log(f"ไม่พบไฟล์น้ำหนัก YOLO: {self.yolo_weights}")
        if not Path(self.classifier_path).exists():
            self.log(f"ไม่พบไฟล์ classifier: {self.classifier_path}")

    def _setup_and_run(self):
        try:
            self._setup_models()
        except Exception as e:
            self.log(f"Model setup failed: {e}")

        cam_index = int(self.camera_index)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            self.log(f"Cannot open camera {cam_index}")
            self.running = False
            return

        self.log("Camera opened")
        self.master.after(30, self._display_loop)

        default_label_map = {0: "with_mask", 1: "without_mask"}

        # ---------- helpers ----------
        def _normalize_label(s: str) -> str:
            s = (s or "").strip().lower()
            s_space = s.replace("_", " ")
            if s in ("without_mask", "no_mask") or s_space in ("without mask", "no mask", "no-mask"):
                return "without_mask"
            if s in ("with_mask",) or s_space in ("with mask", "mask"):
                return "with_mask"
            return s

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def _is_duplicate_event(*args, **kwargs):
            # DEPRECATED: replaced by track-based debouncing
            return False

        def _persist_daily_row(day_str):
            place = self.location_name or "ไม่ระบุ"
            total = self.with_mask_count_today + self.without_mask_count_today
            pct = (self.with_mask_count_today / total * 100.0) if total > 0 else 0.0

            write_header = (not self.daily_csv_path.exists()) or (self.daily_csv_path.stat().st_size == 0)
            with open(self.daily_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "วันที่",
                        "สวมหน้ากากทั้งหมดต่อวัน",
                        "ไม่สวมหน้ากากทั้งหมดต่อวัน",
                        "รวมสวม/ไม่สวมหน้ากากทั้งหมดต่อวัน",
                        "เปอร์เซ็นต์สวมหน้ากากต่อวัน",
                        "สถานที่"
                    ])
                w.writerow([
                    day_str,
                    self.with_mask_count_today,
                    self.without_mask_count_today,
                    total,
                    f"{pct:.2f}",
                    place
                ])

        def _rollover_day_if_needed():
            curr = datetime.now().strftime("%Y-%m-%d")
            if curr != self.today_str:
                try:
                    _persist_daily_row(self.today_str)
                except Exception as e:
                    self.log(f"Persist daily row failed: {e}")
                self.today_str = curr
                self.with_mask_count_today = 0
                self.without_mask_count_today = 0

        def _inc_counter(norm_label):
            _rollover_day_if_needed()
            if norm_label == "with_mask":
                self.with_mask_count_today += 1
                delta_with, delta_without = 1, 0   # <-- เพิ่ม 1
            elif norm_label == "without_mask":
                self.without_mask_count_today += 1
                delta_with, delta_without = 0, 1   # <-- เพิ่ม 1
            else:
                delta_with, delta_without = 0, 0

            if self.show_running_totals:
                total = self.with_mask_count_today + self.without_mask_count_today
                pct = (self.with_mask_count_today / total * 100.0) if total > 0 else 0.0
                self.log(f"วันนี้ ใส่หน้ากาก={self.with_mask_count_today} | ไม่ใส่={self.without_mask_count_today} | รวม={total} | %ใส่={pct:.2f}")

            # [DB] อัปเดตแบบ delta (บวกเพิ่ม)
            try:
                place = self.location_name or "ไม่ระบุ"
                self.db.upsert_daily_delta_async(
                    day_str=self.today_str,
                    location_name=place,
                    with_mask_delta=delta_with,
                    without_mask_delta=delta_without
                )
            except Exception as e:
                self.log(f"DB daily upsert err: {e}")


        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------- ใช้ BGR กับตัวตรวจจับ ----------------
            detections = []
            try:
                if self.face_detector is not None:
                    dets = self.face_detector.detect(frame)  # BGR in
                    for d in dets:
                        x1, y1, x2, y2, conf = d[:5]
                        detections.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'conf': float(conf)})
                elif self.yolo_model is not None:
                    res = self.yolo_model(frame, conf=self.conf_thresh_fixed)[0]  # BGR in
                    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
                    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else []
                    for (b, c) in zip(boxes, confs):
                        x1, y1, x2, y2 = b.astype(int)
                        detections.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'conf': float(c)})
            except Exception as e:
                print("Detection error:", e)

            # แปลงเป็น RGB เพื่อแสดงผล
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------- Tracking update ----------
            if self._tracker is None:
                self._tracker = SimpleByteTracker(
                    iou_threshold=self.iou_threshold, max_age=self.max_age, min_hits=self.min_hits,
                    ema_beta=self.ema_beta, t_on=self.t_on, t_off=self.t_off, margin=self.margin,
                    K=self.K, window_N=self.window_N, num_classes=2
                )
            self._tracker.update(detections)

            # ---------- Classification per visible track + temporal smoothing ----------
            results = []  # list of dict: {tid, box, label, conf}
            for t in self._tracker.outputs():
                x1, y1, x2, y2 = t.box
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(rgb.shape[1]-1, x2); y2 = min(rgb.shape[0]-1, y2)
                face_crop = rgb[y1:y2, x1:x2]

                # default probs (uniform) ใช้กรณีไม่มีหน้าครอป/คลาสสิฟายเออร์
                p_final = np.full((2,), 0.5, dtype=np.float32)

                if self.classifier is not None and face_crop.size != 0 and torch is not None:
                    img = cv2.resize(face_crop, (224, 224)).astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img = (img - mean) / std
                    img = np.transpose(img, (2, 0, 1))
                    tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.classifier(tensor)
                        probs = F.softmax(out, dim=1).cpu().numpy()[0].astype(np.float32)
                        if probs.shape[0] == 2:
                            p_final = probs

                lbl_idx_str, conf = self._tracker.update_probs_and_get_label(t, p_final)
                lbl_idx = int(lbl_idx_str)
                label = ("with_mask" if lbl_idx == 0 else "without_mask")

                results.append({
                    "tid": t.tid, "box": [x1, y1, x2, y2], "label": label, "conf": float(conf)
                })

            # ---------- counting + saving + alert (Track-based) ----------
            now_ts = time.time()
            tid_to_track = {t.tid: t for t in self._tracker.outputs()}

            for item in results:
                tid   = item["tid"]
                box   = item["box"]
                label = _normalize_label(item["label"])
                conf  = float(item["conf"])

                t = tid_to_track.get(tid, None)
                if t is None:
                    continue

                # ========== One-shot counting per track-session ==========
                # นับ "ครั้งเดียว" ต่อ track เมื่อสถานะ (label) ผ่านเงื่อนไขความมั่นใจ (EMA+hysteresis) แล้วเท่านั้น
                if t.counted_label is None:
                    if label in ("with_mask", "without_mask") and conf >= self.t_on:
                        _inc_counter(label)          # <-- count once
                        t.counted_label = label
                        t.counted_ts = now_ts

                # ========== Save images for 'without_mask' (still useful even if already counted) ==========
                if label == "without_mask":
                        if t.counted_label == label and t.last_save_ts == 0.0:
                            self._save_record(rgb, box, label, conf)
                            t.last_save_ts = now_ts

                        if self.alert_enabled:
                            if now_ts - self._last_alert_time >= self.alert_cooldown_sec:
                                self._last_alert_time = now_ts
                                threading.Thread(target=self._play_alert, daemon=True).start()

            # ---------- draw ----------

            display = rgb.copy()
            if self.show_boxes:
                for item in results:
                    x1, y1, x2, y2 = item['box']
                    tag_text = f"ID {item['tid']} | {item['label']} {item['conf']:.2f}"
                    try:
                        norm = (item['label'] or "").lower().replace(" ", "_")
                        if "with_mask" in norm:
                            color_key = 0
                        elif "incorrect" in norm:
                            color_key = 1
                        else:
                            color_key = 2
                        if draw_box_with_label is not None:
                            draw_box_with_label(display, (x1, y1, x2, y2), tag_text, color_key, conf=None)
                        else:
                            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0) if color_key==0 else (0,165,255) if color_key==1 else (0,0,255), 2)
                            cv2.putText(display, tag_text, (x1, max(10, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print("Draw error:", e)


            
            now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display, now_text, (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.current_frame = display

            try:
                if not self.frame_q.empty():
                    try: self.frame_q.get_nowait()
                    except Exception: pass
                self.frame_q.put_nowait(display)
            except Exception:
                pass

            time.sleep(0.02)

        cap.release()
        self.running = False
        self.log("Camera stopped")

    def _display_loop(self):
        try:
            frame = self.frame_q.get_nowait()
            h, w = frame.shape[:2]
            # ใช้ขนาดปัจจุบันของ canvas
            canvas_w = self.canvas.winfo_width()  or int(self.canvas['width'])
            canvas_h = self.canvas.winfo_height() or int(self.canvas['height'])
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = Image.fromarray(frame).resize((new_w, new_h))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self.canvas_img_id, image=self.photo)
            self.canvas.coords(self.canvas_img_id, canvas_w//2, canvas_h//2)
        except queue.Empty:
            pass
        if self.running:
            self.master.after(30, self._display_loop)

    def _on_close(self):
        if self.running:
            if not self._confirm("ปิดโปรแกรม", "ต้องการหยุดกล้องและออกจากโปรแกรมหรือไม่?"):
                return
            self.running = False
            time.sleep(0.1)
        # persist last day to CSV
        try:
            total = self.with_mask_count_today + self.without_mask_count_today
            if total > 0:
                place = self.location_name or "ไม่ระบุ"
                write_header = (not self.daily_csv_path.exists()) or (self.daily_csv_path.stat().st_size == 0)
                pct = (self.with_mask_count_today / total * 100.0) if total > 0 else 0.0
                with open(self.daily_csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow([
                            "วันที่",
                            "สวมหน้ากากทั้งหมดต่อวัน",
                            "ไม่สวมหน้ากากทั้งหมดต่อวัน",
                            "รวมสวม/ไม่สวมหน้ากากทั้งหมดต่อวัน",
                            "เปอร์เซ็นต์สวมหน้ากากต่อวัน",
                            "สถานที่"
                        ])
                    w.writerow([
                        self.today_str,
                        self.with_mask_count_today,
                        self.without_mask_count_today,
                        total,
                        f"{pct:.2f}",
                        place
                    ])
        except Exception as e:
            self.log(f"Persist on close failed: {e}")

        # try:
        #     self.db.stop_worker()
        #     self.db.close()
        # except Exception:
        #     pass

        # ปิดเสียง/mixer ให้เรียบร้อย
        try:
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        except Exception:
            pass

        # FIX: ล้าง ref ของ records_tree กัน callback ไปแตะ widget หลัง destroy
        try:
            self.db.stop_worker()
            self.db.close()
        except Exception:
            pass
        self.records_tree = None

        self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = MaskDetectorApp(root)
    root.mainloop()
