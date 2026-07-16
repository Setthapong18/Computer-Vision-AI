# 🎯 AI Computer Vision — Real-Time Person Detection & Tracking

> ระบบตรวจจับและติดตามบุคคลแบบ Real-Time ด้วย YOLOv8 + DeepSORT  
> **โปรเจคกลุ่ม | วิชา Computer Vision | มหาวิทยาลัย**

> **v2.0 — Senior-Level Optimized Edition**

---

## 📌 ภาพรวมโปรเจค

ระบบนี้พัฒนาขึ้นเพื่อตรวจจับและติดตามบุคคลในวิดีโอโดยอัตโนมัติ โดยผสานเทคโนโลยีสองตัวหลักเข้าด้วยกัน:

- **YOLOv8** — โมเดล Deep Learning สำหรับตรวจจับวัตถุ (Object Detection) แบบ Real-Time
- **DeepSORT** — อัลกอริทึมสำหรับติดตามวัตถุข้ามเฟรม (Multi-Object Tracking)

ระบบสามารถนับจำนวนบุคคลในแต่ละเฟรม, กำหนด Tracking ID ให้แต่ละคน, และบันทึกผลลัพธ์ออกมาเป็นไฟล์วิดีโอได้โดยอัตโนมัติ

---

## ✨ ฟีเจอร์หลัก

| ฟีเจอร์ | รายละเอียด |
|---|---|
| 🔍 Person Detection | YOLOv8 Conf≥45%, ตรวจจับคนที่ถูกบังได้มากขึ้น |
| 🆔 Unique-Color Tracking | สีเฉพาะต่อ Track ID ผ่าน HSV colormap |
| 🛤️ Trajectory Trail | แสดงเส้นทางการเดินย้อนหลัง 30 เฟรม |
| 🔲 L-Shaped Corner Box | Bounding box สไตล์ professional |
| 🔢 Real-Time Stats | จำนวนคน ปัจจุบัน + รวมทั้งหมด (unique) |
| ⚡ FPS Counter | วัด FPS จริงแบบ rolling average |
| ⏸️ Pause / Resume | กด `p` เพื่อหยุดชั่วคราว |
| 🎬 Smart Video Export | บันทึก FPS และ resolution จริงจากวิดีโอต้นฉบับ |
| 🇹🇭 Thai Stats Panel | Overlay ภาษาไทยกึ่งโปร่งใสมุมซ้ายบน |

---

## 🛠️ เทคโนโลยีที่ใช้

```
YOLOv8 (Ultralytics)         — Object Detection Model
DeepSORT (deep_sort_realtime) — Multi-Object Tracking
OpenCV (cv2)                  — Video Processing & Display
Pillow (PIL)                  — Thai Text Rendering on Frames
NumPy                         — Numerical Array Operations
CUDA (GPU)                    — Accelerated Model Training
```

---

## 📁 โครงสร้างโปรเจค

```
AI Computer Vision/
│
├── detect_video.py         # ไฟล์หลัก: ตรวจจับและติดตามบุคคลในวิดีโอ
├── train.py                # ไฟล์เทรนโมเดล YOLOv8 ด้วย Custom Dataset
├── best.pt                 # โมเดล YOLOv8 ที่เทรนแล้ว (ไม่รวมใน repo)
├── README.md               # เอกสารอธิบายโปรเจค
│
├── datasets/
│   └── data.yaml           # ไฟล์กำหนด dataset configuration
│
├── Test/
│   └── test5.mp4           # ไฟล์วิดีโอทดสอบ (ไม่รวมใน repo)
│
└── runs/train/
    └── yolov8m_custom_optimized/  # ผลลัพธ์จากการเทรน
```

---

## ⚙️ การติดตั้ง

### 1. Clone โปรเจค
```bash
git clone https://github.com/<your-username>/AI-Computer-Vision.git
cd AI-Computer-Vision
```

### 2. ติดตั้ง Dependencies
```bash
pip install ultralytics deep-sort-realtime opencv-python pillow numpy
```

> **หมายเหตุ:** ต้องติดตั้ง [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) หากต้องการใช้ GPU ในการเทรน

### 3. เตรียมโมเดล
วางไฟล์ `best.pt` (โมเดลที่เทรนแล้ว) ไว้ใน root directory

---

## 🚀 วิธีใช้งาน

### การตรวจจับและติดตามจากวิดีโอ

1. วางไฟล์วิดีโอที่ต้องการไว้ใน `Test/`
2. แก้ไข `video_path` ใน `detect_video.py`:
   ```python
   video_path = "Test/your_video.mp4"
   ```
3. รันสคริปต์:
   ```bash
   python detect_video.py
   ```
4. ผลลัพธ์จะถูกบันทึกเป็น `output_<ชื่อวิดีโอ>_<timestamp>.mp4`
5. กด `q` เพื่อหยุดการแสดงผล

### การเทรนโมเดลใหม่

1. เตรียม Dataset และไฟล์ `datasets/data.yaml`
2. รันสคริปต์เทรน:
   ```bash
   python train.py
   ```
3. ผลลัพธ์จะอยู่ใน `runs/train/yolov8m_custom_optimized/`

---

## 🔧 การปรับแต่งพารามิเตอร์

### detect_video.py

| พารามิเตอร์ | ค่าใหม่ (Optimized) | คำอธิบาย |
|---|---|---|
| `CONF_THRESHOLD` | `0.45` ↓ จาก 0.65 | เพิ่ม recall จับคนที่ถูกบังได้มากขึ้น |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `DEEPSORT_MAX_AGE` | `70` ↑ จาก 50 | ติดตามได้นานขึ้นเมื่อหายชั่วคราว |
| `DEEPSORT_N_INIT` | `3` ↑ จาก 2 | ลด false track เพิ่มความแม่นยำ |
| `DEEPSORT_MAX_COS_DIST` | `0.5` ↓ จาก 0.7 | Re-ID เข้มงวดขึ้น ลด ID switch |
| `TRAIL_LENGTH` | `30` | ความยาวเส้น trajectory |

### train.py

| พารามิเตอร์ | ค่าใหม่ (Optimized) | คำอธิบาย |
|---|---|---|
| `optimizer` | `AdamW` | เสถียรกว่า SGD สำหรับ dataset ขนาดกลาง |
| `lr0` | `0.001` | Initial learning rate |
| `epochs` | `100` | มี Early Stopping (patience=20) |
| `batch` | `16` | ปรับตาม VRAM (4GB→8, 8GB→16) |
| `amp` | `True` | Mixed Precision ประหยัด VRAM ~40% |
| `mosaic` | `1.0` | Augmentation รวม 4 ภาพ |
| `mixup` | `0.15` | Augmentation ผสม 2 ภาพ |
| `patience` | `20` | Early Stopping epochs |

---

## 🎬 ตัวอย่างผลลัพธ์

ระบบจะแสดงผลลัพธ์บนเฟรมวิดีโอดังนี้:

- **กรอบ L-shaped corners** สีเฉพาะต่อแต่ละ Track ID
- **Trajectory trail** เส้นทางการเดินย้อนหลัง 30 เฟรม
- **Label** `ID:X  0.87%` แสดง ID และ confidence ต่อคน
- **Stats panel** มุมซ้ายบน: คนปัจจุบัน / รวมทั้งหมด / FPS / เฟรม

### Controls
| ปุ่ม | การทำงาน |
|---|---|
| `q` | ออกจากโปรแกรม |
| `p` | Pause / Resume |

---

## 👥 ทีมพัฒนา

โปรเจคนี้พัฒนาโดยทีมนักศึกษา ภายใต้วิชา Computer Vision

<!-- เพิ่มชื่อสมาชิกทีมที่นี่ -->

---

## 📄 License

This project is for educational purposes only.

---

## 🔗 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DeepSORT Real-time](https://github.com/levan92/deep_sort_realtime)
- [OpenCV Documentation](https://docs.opencv.org/)
