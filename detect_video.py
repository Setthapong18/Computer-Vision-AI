"""
detect_video.py — Optimized Person Detection & Tracking
============================================================
YOLOv8 + DeepSORT | Real-Time Person Tracking
ปรับค่าใน CONFIG section ด้านล่าง

Controls:
  q → ออกจากโปรแกรม
  p → Pause / Resume
"""

import cv2
import numpy as np
import time
import os
from collections import defaultdict, deque
from datetime import datetime

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# ⚙️  CONFIG — ปรับค่าที่นี่ได้เลย
# ============================================================
VIDEO_PATH             = "Test/test5.mp4"
MODEL_PATH             = "yolov8m.pt"
OUTPUT_DIR             = "output"
FONT_PATH              = "C:/Windows/Fonts/tahoma.ttf"

# Detection thresholds
CONF_THRESHOLD         = 0.25   # ลดลงเพื่อจับคนไกลและคนเล็กได้มากขึ้น
IOU_THRESHOLD          = 0.40   # NMS IoU threshold
INFERENCE_IMGSZ        = 1280   # เพิ่มจาก 640 → จับ object เล็ก/ไกลได้ดีขึ้น
CUDA_DEVICE            = 0      # GPU device index (0 = GTX 1060)

# Visualization
TRAIL_LENGTH           = 30     # ความยาวเส้น trajectory (เฟรม)

# DeepSORT — tuned for person tracking
DEEPSORT_MAX_AGE       = 70     # เฟรมสูงสุดที่ track อยู่แม้ไม่เห็น
DEEPSORT_N_INIT        = 3      # เฟรมขั้นต่ำก่อนยืนยัน track ใหม่
DEEPSORT_MAX_COS_DIST  = 0.5   # Re-ID threshold (ต่ำ = เข้มงวด)
DEEPSORT_NN_BUDGET     = 100    # Feature budget ต่อ track


# ============================================================
# 🎨  DRAWING HELPERS
# ============================================================

def get_track_color(track_id) -> tuple:
    """สร้างสีเฉพาะต่อ Track ID (ไม่ซ้ำกัน) ผ่าน HSV colormap"""
    hue = int((int(track_id) * 47) % 180)
    hsv = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_ratio=0.25):
    """
    วาด Bounding Box แบบ L-shaped corners (professional look)
    แทนกล่องธรรมดา
    """
    w, h = x2 - x1, y2 - y1
    clen = int(min(w, h) * corner_ratio)   # ความยาว corner ตามขนาด bbox

    t = thickness + 1  # corners หนากว่า outline หลัก

    # กล่องหลัก (บาง)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # มุมบนซ้าย
    cv2.line(frame, (x1, y1), (x1 + clen, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + clen), color, t)
    # มุมบนขวา
    cv2.line(frame, (x2, y1), (x2 - clen, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + clen), color, t)
    # มุมล่างซ้าย
    cv2.line(frame, (x1, y2), (x1 + clen, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - clen), color, t)
    # มุมล่างขวา
    cv2.line(frame, (x2, y2), (x2 - clen, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - clen), color, t)


def draw_trajectory(frame, trail: deque, color: tuple):
    """วาดเส้น trajectory ที่ค่อยๆ จางตามความเก่า"""
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        alpha      = i / n   # 0 = เก่า (จาง) → 1 = ใหม่ (เข้ม)
        fade_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pts[i - 1], pts[i], fade_color, 2, cv2.LINE_AA)


def draw_label(frame, text: str, x: int, y: int, color: tuple, font_scale=0.45):
    """วาด label พร้อม background สีดำกึ่งโปร่งใส"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, 1)

    pad_x, pad_y = 4, 3
    bx1, by1 = x - pad_x, y - th - pad_y
    bx2, by2 = x + tw + pad_x, y + bl + pad_y

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)


def draw_stats_panel(frame, info_font, title_font,
                     person_count: int, unique_total: int,
                     fps: float, frame_idx: int) -> np.ndarray:
    """
    วาด semi-transparent stats panel มุมบนซ้าย
    ใช้ PIL เพื่อรองรับภาษาไทย
    """
    panel_w, panel_h = 270, 125
    margin = 12

    # วาด background ด้วย OpenCV ก่อน (เร็ว)
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (margin, margin),
                  (margin + panel_w, margin + panel_h),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # แปลงเฉพาะส่วน panel เป็น PIL (ประหยัดกว่าแปลงทั้งเฟรม)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    lx = margin + 12
    draw.text((lx, margin + 8),  f"👤 คนปัจจุบัน : {person_count}",    font=info_font,  fill=(120, 255, 120))
    draw.text((lx, margin + 35), f"📊 รวมทั้งหมด : {unique_total} คน", font=info_font,  fill=(120, 200, 255))
    draw.text((lx, margin + 62), f"⚡ FPS         : {fps:.1f}",          font=info_font,  fill=(255, 210, 80))
    draw.text((lx, margin + 89), f"🎬 เฟรม        : {frame_idx}",        font=info_font,  fill=(200, 200, 200))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_stats_panel_cv(frame, person_count, unique_total, fps, frame_idx):
    """Fallback stats panel ใช้ OpenCV (กรณีไม่มีฟอนต์ไทย)"""
    margin = 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + 270, margin + 125), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    font, s, lx = cv2.FONT_HERSHEY_SIMPLEX, 0.52, margin + 12
    cv2.putText(frame, f"Current  : {person_count} persons",   (lx, margin + 26),  font, s, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total    : {unique_total} persons",   (lx, margin + 52),  font, s, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS      : {fps:.1f}",                (lx, margin + 78),  font, s, (255, 210, 80),  1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame    : {frame_idx}",              (lx, margin + 104), font, s, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# 🚀  MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── โหลดโมเดล ──────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        device_str = f"cuda:{CUDA_DEVICE}"
        gpu_name   = torch.cuda.get_device_name(CUDA_DEVICE)
        vram_gb    = torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1024**3
        print(f"[INFO] GPU  : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device_str = "cpu"
        print("[WARN] GPU ไม่พบ — รันบน CPU (ช้ากว่า GPU มาก)")
    print(f"[INFO] กำลังโหลดโมเดล: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ── DeepSORT tracker ────────────────────────────────────
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        max_cosine_distance=DEEPSORT_MAX_COS_DIST,
        nn_budget=DEEPSORT_NN_BUDGET,
    )

    # ── เปิดวิดีโอ ──────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] ไม่สามารถเปิดวิดีโอ: {VIDEO_PATH}")
        return

    # อ่าน properties จริงจากวิดีโอ (ไม่ hard-code)
    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] วิดีโอ : {frame_w}x{frame_h} @ {src_fps:.1f} FPS | {total_f} เฟรม")

    # ── เตรียมไฟล์ output ───────────────────────────────────
    stem       = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = os.path.join(OUTPUT_DIR, f"output_{stem}_{ts}.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(out_path, fourcc, src_fps, (frame_w, frame_h))

    # ── โหลดฟอนต์ภาษาไทย ────────────────────────────────────
    using_thai = False
    info_font  = None
    title_font = None
    for fp in [FONT_PATH,
               "C:/Windows/Fonts/THSarabunNew.ttf",
               "C:/Windows/Fonts/arial.ttf"]:
        try:
            info_font  = ImageFont.truetype(fp, 18)
            title_font = ImageFont.truetype(fp, 22)
            using_thai = True
            print(f"[INFO] โหลดฟอนต์สำเร็จ: {fp}")
            break
        except Exception:
            continue
    if not using_thai:
        print("[WARN] ไม่พบฟอนต์ — ใช้ OpenCV font แทน")

    # ── State tracking ──────────────────────────────────────
    trajectories  = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    track_confs   = {}              # confidence ล่าสุดต่อ track
    unique_ids    = set()           # unique track IDs ทั้งหมด
    fps_history   = deque(maxlen=30)
    frame_idx     = 0
    paused        = False

    print("[INFO] เริ่มประมวลผล...")
    print("       กด 'q' เพื่อออก | 'p' เพื่อ Pause/Resume\n")

    try:
        while cap.isOpened():
            # ── Keyboard control ────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] ผู้ใช้กด 'q' — หยุดประมวลผล")
                break
            if key == ord("p"):
                paused = not paused
                status = "หยุดชั่วคราว" if paused else "เล่นต่อ"
                print(f"[INFO] {status}")

            if paused:
                continue

            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            display   = frame.copy()

            # ══════════════════════════════════════════════
            # 🔍  YOLO Detection
            # ══════════════════════════════════════════════
            results = model(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=INFERENCE_IMGSZ,
                classes=[0],      # ตรวจจับเฉพาะ class 0 = "person"
                device=device_str,# ใช้ GPU ถ้ามี
                augment=True,     # Test-Time Augmentation ช่วยจับคนไกล/เล็ก
                verbose=False,
            )

            detections = []
            for result in results:
                for box in result.boxes:
                    conf          = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h          = x2 - x1, y2 - y1
                    # รูปแบบที่ DeepSORT ต้องการ: [x, y, w, h], conf, class_id
                    detections.append(([x1, y1, w, h], conf, 0))

            # ══════════════════════════════════════════════
            # 🎯  DeepSORT Tracking
            # ══════════════════════════════════════════════
            tracked = tracker.update_tracks(detections, frame=frame)

            person_count = 0
            for track in tracked:
                if not track.is_confirmed():
                    continue

                track_id       = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Clamp ให้อยู่ในขอบเฟรม
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(frame_w - 1, x2);  y2 = min(frame_h - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue   # bbox เสียหาย ข้าม

                color       = get_track_color(track_id)
                cx, cy      = (x1 + x2) // 2, (y1 + y2) // 2
                person_count += 1
                unique_ids.add(track_id)

                # อัปเดต confidence (จาก DeepSORT attribute หรือ cache)
                try:
                    conf_val = float(track.det_conf) if track.det_conf is not None else track_confs.get(track_id, 0.0)
                except Exception:
                    conf_val = track_confs.get(track_id, 0.0)
                track_confs[track_id] = conf_val

                # อัปเดต trajectory
                trajectories[track_id].append((cx, cy))

                # ── วาดผลลัพธ์ ──────────────────────────
                draw_trajectory(display, trajectories[track_id], color)
                draw_corner_box(display, x1, y1, x2, y2, color, thickness=2)

                label = f"ID:{track_id}  {conf_val:.0%}"
                draw_label(display, label, x1, max(y1 - 6, 12), color)

            # ══════════════════════════════════════════════
            # 📊  Stats Panel
            # ══════════════════════════════════════════════
            elapsed = time.perf_counter() - t0
            fps_history.append(1.0 / max(elapsed, 1e-6))
            avg_fps = sum(fps_history) / len(fps_history)

            if using_thai:
                display = draw_stats_panel(
                    display, info_font, title_font,
                    person_count, len(unique_ids), avg_fps, frame_idx
                )
            else:
                draw_stats_panel_cv(display, person_count, len(unique_ids), avg_fps, frame_idx)

            # ══════════════════════════════════════════════
            # 💾  บันทึก & แสดงผล
            # ══════════════════════════════════════════════
            writer.write(display)
            cv2.imshow("Person Tracking  |  YOLOv8 + DeepSORT  |  q=Quit  p=Pause", display)

            # Progress log ทุก 60 เฟรม
            if frame_idx % 60 == 0:
                pct = (frame_idx / total_f * 100) if total_f > 0 else 0
                print(f"  เฟรม {frame_idx:>5}/{total_f}  ({pct:4.1f}%)  "
                      f"FPS:{avg_fps:5.1f}  คนปัจจุบัน:{person_count:3d}  "
                      f"รวมทั้งหมด:{len(unique_ids):3d}")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*55}")
        print(f"  [DONE] บันทึกวิดีโอแล้ว : {out_path}")
        print(f"  [STAT] เฟรมที่ประมวลผล : {frame_idx} เฟรม")
        print(f"  [STAT] บุคคล (unique)  : {len(unique_ids)} คน")
        print(f"{'='*55}\n")


# ============================================================
if __name__ == "__main__":
    main()