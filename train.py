"""
train.py — Optimized YOLOv8 Training Script
============================================================
Senior-level configuration สำหรับ Person Detection
ปรับค่าใน CONFIG section ด้านล่างก่อนเริ่มเทรน
"""

from ultralytics import YOLO
import torch

# ============================================================
# 🖥️  แสดงข้อมูล GPU
# ============================================================
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[INFO] GPU   : {gpu_name}")
    print(f"[INFO] VRAM  : {vram_gb:.1f} GB")
else:
    print("[WARN] ไม่พบ GPU — จะใช้ CPU (ช้ามาก ไม่แนะนำ)")

# ============================================================
# ⚙️  CONFIG — ปรับค่าที่นี่ก่อนเทรน
# ============================================================
MODEL_WEIGHTS = "yolov8m.pt"       # yolov8n/s/m/l/x  (n=เล็ก/เร็ว ↔ x=ใหญ่/แม่น)
DATA_YAML     = "datasets/data.yaml"
PROJECT       = "runs/train"
NAME          = "yolov8m_optimized"
IMGSZ         = 640
EPOCHS        = 100                # มี Early Stopping ดักไว้ ไม่ต้องกังวลเกิน
DEVICE        = "cuda"             # "cuda" หรือ "cpu"

# ปรับ BATCH ตาม VRAM:
#   4 GB  → 8
#   6 GB  → 12
#   8 GB  → 16
#   12 GB → 24
#   16 GB → 32
BATCH = 16

# ============================================================
# 🚀  LOAD MODEL
# ============================================================
print(f"\n[INFO] โหลดโมเดล: {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)

# ============================================================
# 🏋️  TRAIN
# ============================================================
print(f"[INFO] เริ่มเทรน | epochs={EPOCHS} | batch={BATCH} | imgsz={IMGSZ}\n")

train_results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    device=DEVICE,
    batch=BATCH,
    workers=4,            # Parallel data loading (เพิ่มจาก 0 → เร็วขึ้นมาก)
    project=PROJECT,
    name=NAME,

    # ── Optimizer ──────────────────────────────────────────
    optimizer="AdamW",    # ดีกว่า SGD default สำหรับ dataset ขนาดกลาง
    lr0=0.001,            # Initial learning rate
    lrf=0.01,             # Final LR = lr0 * lrf (Cosine decay)
    momentum=0.937,
    weight_decay=0.0005,

    # ── LR Warmup ──────────────────────────────────────────
    warmup_epochs=3,      # ป้องกัน gradient explosion ช่วงแรก
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # ── Loss Weights ───────────────────────────────────────
    box=7.5,              # Box regression loss
    cls=0.5,              # Classification loss
    dfl=1.5,              # Distribution Focal Loss

    # ── Data Augmentation ──────────────────────────────────
    # Color jitter: ป้องกัน overfit สี/แสง
    hsv_h=0.015,          # Hue shift ±1.5%
    hsv_s=0.7,            # Saturation ±70%
    hsv_v=0.4,            # Value (brightness) ±40%
    # Geometric: ทำให้โมเดล robust ต่อตำแหน่ง/มุม
    degrees=5.0,          # หมุน ±5 องศา
    translate=0.1,        # เลื่อน ±10%
    scale=0.5,            # ซูม 50–150%
    shear=0.0,
    perspective=0.0,
    flipud=0.0,           # ไม่พลิกแนวตั้ง (คนไม่ยืนหัวกลับ)
    fliplr=0.5,           # พลิกแนวนอน 50%
    # Advanced augmentation:
    mosaic=1.0,           # รวม 4 ภาพ → context หลากหลาย (แนะนำมาก)
    mixup=0.15,           # ผสม 2 ภาพ → regularization
    copy_paste=0.1,       # ตัด-แปะ object ข้ามภาพ
    close_mosaic=10,      # ปิด mosaic ใน 10 epochs สุดท้าย → fine-tune ดีขึ้น

    # ── Training Tricks ────────────────────────────────────
    amp=True,             # Mixed Precision FP16 (ประหยัด VRAM ~40%, เร็วขึ้น)
    cache="ram",          # Cache dataset ใน RAM (เร็วขึ้นมาก ต้องการ RAM พอ)
    patience=20,          # Early Stopping: หยุดถ้าไม่ดีขึ้น 20 epochs ติดกัน
    save_period=5,        # บันทึก checkpoint ทุก 5 epochs
    nbs=64,               # Nominal batch size → auto-scale LR ตาม batch จริง
    label_smoothing=0.0,
    dropout=0.0,

    # ── Output ─────────────────────────────────────────────
    val=True,             # Validate หลังทุก epoch
    plots=True,           # บันทึก training curves, confusion matrix
    verbose=True,
)

# ============================================================
# 📊  VALIDATE & PRINT METRICS
# ============================================================
print("\n" + "=" * 50)
print("[INFO] กำลัง Validate โมเดลที่เทรนแล้ว...")
print("=" * 50)

metrics = model.val()

print(f"\n{'─'*50}")
print(f"  mAP@50     : {metrics.box.map50:.4f}  ({metrics.box.map50 * 100:.2f}%)")
print(f"  mAP@50-95  : {metrics.box.map:.4f}   ({metrics.box.map * 100:.2f}%)")
print(f"  Precision  : {metrics.box.mp:.4f}")
print(f"  Recall     : {metrics.box.mr:.4f}")
print(f"{'─'*50}")
print(f"  โมเดลที่ดีที่สุดอยู่ที่: {PROJECT}/{NAME}/weights/best.pt")
print(f"{'─'*50}\n")
