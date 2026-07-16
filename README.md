# 🎯 AI Computer Vision — Real-Time Person Detection & Tracking

> ระบบตรวจจับและติดตามบุคคลแบบ Real-Time ด้วย YOLOv8 + DeepSORT  
> **โปรเจคกลุ่ม | วิชา Computer Vision | มหาวิทยาลัย**

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
| 🔍 Person Detection | ตรวจจับบุคคลในวิดีโอด้วย YOLOv8 (Confidence ≥ 65%) |
| 🆔 Object Tracking | ติดตามแต่ละคนด้วย Unique ID ผ่าน DeepSORT |
| 🔢 People Count | แสดงจำนวนบุคคลแบบ Real-Time บนเฟรม |
| 📊 Detection % | แสดงเปอร์เซ็นต์ความมั่นใจในการตรวจจับ |
| 🎬 Video Export | บันทึกวิดีโอผลลัพธ์พร้อม Timestamp อัตโนมัติ |
| 🇹🇭 Thai UI | แสดงข้อความภาษาไทยบนเฟรม (ผ่าน Pillow + Tahoma Font) |

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

| พารามิเตอร์ | ค่าปัจจุบัน | คำอธิบาย |
|---|---|---|
| `conf` | `0.65` | Confidence threshold สำหรับ YOLO |
| `iou` | `0.5` | IoU threshold สำหรับ NMS |
| `max_age` | `50` | จำนวนเฟรมสูงสุดที่ DeepSORT ยังคง track ไว้ |
| `n_init` | `2` | จำนวนเฟรมขั้นต่ำเพื่อยืนยัน track ใหม่ |
| `max_cosine_distance` | `0.7` | ระยะทาง cosine สูงสุดสำหรับ re-identification |

### train.py

| พารามิเตอร์ | ค่าปัจจุบัน | คำอธิบาย |
|---|---|---|
| `epochs` | `50` | จำนวนรอบการเทรน |
| `imgsz` | `640` | ขนาดภาพ input |
| `batch` | `6` | Batch size |
| `device` | `cuda` | ใช้ GPU ในการเทรน |

---

## 🎬 ตัวอย่างผลลัพธ์

ระบบจะแสดงผลลัพธ์บนเฟรมวิดีโอดังนี้:

- **กรอบสีเขียว** รอบแต่ละบุคคลที่ตรวจพบ
- **ID** ประจำแต่ละคนที่กำหนดโดย DeepSORT
- **จำนวนคน** แสดงบนมุมซ้ายบน (ภาษาไทย)
- **เปอร์เซ็นต์ความมั่นใจ** แสดงใต้กรอบแต่ละคน

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
