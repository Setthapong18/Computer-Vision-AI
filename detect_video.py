# นำเข้าไลบรารีที่จำเป็นสำหรับประมวลผลวิดีโอ ตรวจจับ และติดตามวัตถุ
import cv2                  # ไลบรารี OpenCV สำหรับการประมวลผลวิดีโอและภาพ
import numpy as np          # ไลบรารีสำหรับการคำนวณทางตัวเลขและจัดการอาร์เรย์
from ultralytics import YOLO  # โมเดลตรวจจับวัตถุ YOLOv8 จาก Ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort  # อัลกอริทึมติดตามวัตถุแบบ Deep Learning
from PIL import Image, ImageDraw, ImageFont  # ไลบรารีสำหรับการจัดการภาพและข้อความขั้นสูง
from datetime import datetime  # สำหรับสร้างชื่อไฟล์วิดีโอที่ไม่ซ้ำกัน
import os  # เพื่อการจัดการชื่อไฟล์

# ====== การตั้งค่าโมเดลและการกำหนดค่า ======
# เริ่มต้นโมเดล YOLO สำหรับตรวจจับวัตถุ
model = YOLO("best.pt")  

# กำหนดค่า DeepSORT tracker สำหรับติดตามวัตถุระหว่างเฟรมวิดีโอ
tracker = DeepSort(
    max_age=50,          # เพิ่มระยะเวลาการติดตามวัตถุ
    n_init=2,            # ลดจำนวนเฟรมที่ใช้ยืนยันการติดตาม
    max_cosine_distance=0.7  # ระยะความคล้ายสูงสุดในการติดตาม
)

# ====== การกำหนดค่าอินพุตวิดีโอ ======
# ระบุเส้นทางไปยังไฟล์วิดีโอที่ต้องการประมวลผล
video_path = "Test/test5.mp4"

# สร้างชื่อไฟล์เอาท์พุตจากชื่อไฟล์อินพุต
input_filename = os.path.basename(video_path)
input_filename_without_ext = os.path.splitext(input_filename)[0]
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = f"output_{input_filename_without_ext}_{current_time}.mp4"

# เปิดวิดีโอสำหรับการอ่านและประมวลผล
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าสามารถเปิดวิดีโอได้หรือไม่
if not cap.isOpened():
    print("ข้อผิดพลาด: ไม่สามารถเปิดวิดีโอได้")
    exit()

# ====== การตั้งค่าการบันทึกวิดีโอ ======
# กำหนดคุณสมบัติของวิดีโอที่จะบันทึก
# ใช้ขนาดและ fps คงเดิม
frame_width = 640
frame_height = 360
fps = 30  # ใช้ค่า fps คงที่

# สร้าง VideoWriter object สำหรับบันทึกวิดีโอ
# ใช้ codec H264 ซึ่งเป็นมาตรฐานและเข้ากันได้กว้าง
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ====== การเตรียมฟอนต์ภาษาไทย ======
# ระบุเส้นทางไปยังฟอนต์ภาษาไทย
font_path = "C:/Windows/Fonts/tahoma.ttf"

try:
    # สร้างฟอนต์ขนาดต่างๆ สำหรับข้อความ
    title_font = ImageFont.truetype(font_path, 32)  # ฟอนต์สำหรับหัวข้อ
    info_font = ImageFont.truetype(font_path, 20)   # ฟอนต์สำหรับข้อมูลเพิ่มเติม
    id_font = ImageFont.truetype(font_path, 16)     # ฟอนต์สำหรับ ID
    using_thai_font = True
except Exception as e:
    print(f"ข้อผิดพลาด: ไม่สามารถโหลดฟอนต์ภาษาไทยได้ - {e}")
    print("จะใช้ข้อความภาษาอังกฤษแทน")
    using_thai_font = False

# ====== ลูปประมวลผลวิดีโอหลัก ======
try:
    while cap.isOpened():
        # อ่านเฟรมปัจจุบันจากวิดีโอ
        ret, original_frame = cap.read()
        
        # หากอ่านเฟรมไม่สำเร็จ (อาจเป็นเพราะจบวิดีโอ) ให้ออกจากลูป
        if not ret:
            break
        
        # ปรับขนาดเฟรมให้ตรงกับขนาดที่กำหนด
        frame = cv2.resize(original_frame, (frame_width, frame_height))
        
        # สร้างสำเนาเฟรมสำหรับการแสดงผล
        display_frame = frame.copy()
        
        # ====== ตรวจจับวัตถุด้วย YOLO ======
        results = model(frame, conf=0.65, iou=0.5, imgsz=640)
        
        # เตรียมลิสต์สำหรับเก็บข้อมูลการตรวจจับ
        detections = []
        
        # ตัวแปรนับจำนวนวัตถุและคนที่ตรวจพบ
        total_detections = 0
        person_detections = 0
        
        # ====== ประมวลผลข้อมูลจาก YOLO ======
        for result in results:
            for box in result.boxes:
                # ดึงข้อมูลสำคัญจากกล่องตรวจจับ
                class_id = int(box.cls[0])  # รหัสหมวดหมู่วัตถุ
                conf = float(box.conf[0])   # ค่าความเชื่อมั่น
                
                total_detections += 1
                
                # กรองเฉพาะการตรวจจับ "คน" 
                if class_id == 0 and conf > 0.65:
                    # แปลงพิกัดกล่องตรวจจับ
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # เพิ่มข้อมูลการตรวจจับในรูปแบบที่ DeepSORT ต้องการ
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                    
                    person_detections += 1
        
        # คำนวณเปอร์เซ็นต์การตรวจจับคน
        person_percentage = (person_detections / total_detections * 100) if total_detections > 0 else 0
        
        # ====== ติดตามวัตถุด้วย DeepSORT ======
        tracked_objects = tracker.update_tracks(detections, frame=display_frame)
        
        # ตัวแปรนับจำนวนคนที่ติดตามได้
        person_count = 0
        
        # ====== วาดผลลัพธ์การติดตามบนเฟรม ======
        for track in tracked_objects:
            # ข้ามการติดตามที่ยังไม่มั่นใจ
            if not track.is_confirmed():
                continue
            
            # ดึงข้อมูลการติดตาม
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            person_count += 1
            
            # วาดกรอบรอบคนที่ตรวจพบ
            color = (0, 255, 0)  # สีเขียวสำหรับการติดตามที่ยืนยันแล้ว
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # เขียน ID และเปอร์เซ็นต์ความมั่นใจของการติดตาม
            cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(display_frame, f"{person_percentage:.2f}%", (x1, y2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # ====== เพิ่มข้อความภาษาไทย ======
        if using_thai_font:
            # แปลงเฟรม OpenCV เป็นรูปภาพ PIL
            pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # เขียนข้อความหัวข้อ
            draw.text((20, 20), f"จำนวนคน: {person_count}", font=title_font, fill=(0, 0, 255))
            draw.text((20, 60), f"เปอร์เซ็นต์คน: {person_percentage:.2f}%", font=info_font, fill=(0, 0, 255))
            
            # แปลงกลับเป็นเฟรม OpenCV
            display_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # ใช้ OpenCV เขียนข้อความภาษาอังกฤษ
            cv2.putText(display_frame, f"People Count: {person_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Person Percentage: {person_percentage:.2f}%", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # เขียนเฟรมลงในวิดีโอที่กำลังบันทึก
        out.write(display_frame)
        
        # แสดงเฟรมในหน้าต่าง
        window_title = "การติดตามคนด้วย YOLOv8" if using_thai_font else "Person Tracking with YOLOv8"
        cv2.imshow(window_title, display_frame)
        
        # ตรวจสอบการกดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ====== ทำความสะอาดและปิดทรัพยากร ======
    # ปิดการอ่านวิดีโอ
    cap.release()
    
    # ปิดการบันทึกวิดีโอ
    out.release()
    
    # ปิดหน้าต่างแสดงผล
    cv2.destroyAllWindows()
    
    # พิมพ์ข้อความแจ้งเส้นทางไฟล์วิดีโอที่บันทึก
    print(f"บันทึกวิดีโอเสร็จสิ้นที่: {output_video_path}")