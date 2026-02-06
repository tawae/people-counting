import cv2
import numpy as np
from ultralytics import YOLO
from sort import * # Thư viện SORT để tracking ID
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Khởi tạo các biến toàn cục
in_count = 0
out_count = 0
cap = None
model = None
tracker = None
line_coords = None
stop_tracking = False

def is_point_above_line(point, line_start, line_end):
    """Kiểm tra xem một điểm nằm phía trên hay phía dưới đường kẻ"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    # Công thức tính vị trí tương đối của điểm so với đường thẳng
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def start_tracking_process():
    global in_count, out_count, cap, model, tracker, stop_tracking, line_coords
    
    # Tải mô hình YOLOv8
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Đọc tọa độ đường kẻ từ file (giả sử file txt có dạng np.array)
    # Trong clip, tác giả lấy tọa độ từ Roboflow
    line_start = (173, 372) # Ví dụ
    line_end = (781, 369)   # Ví dụ
    
    track_positions = {} # Lưu vị trí trước đó để biết hướng di chuyển

    while cap.isOpened() and not stop_tracking:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.3: # Chỉ lấy class 'person'
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Cập nhật tracker
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Logic đếm dựa trên việc cắt qua đường kẻ
            current_pos = is_point_above_line(midpoint, line_start, line_end)
            
            if obj_id in track_positions:
                prev_pos = track_positions[obj_id]
                
                if prev_pos < 0 and current_pos >= 0:
                    in_count += 1
                elif prev_pos > 0 and current_pos <= 0:
                    out_count += 1
            
            track_positions[obj_id] = current_pos

            # Vẽ bounding box và ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Vẽ đường phân cách
        cv2.line(frame, line_start, line_end, (255, 0, 0), 3)
        
        # Hiển thị số lượng lên màn hình
        cv2.putText(frame, f"In: {in_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Out: {out_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("People Counting System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giữ cửa sổ lại sau khi hết video
    if not stop_tracking:
        cv2.putText(frame, "Video Ended. Press any key to close.", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("People Counting System", frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# --- Giao diện Tkinter ---
def open_file():
    global cap
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        start_tracking_process()  # Auto-start ngay khi chọn video

root = tk.Tk()
root.title("Hệ thống đếm người AI")
root.geometry("400x300")

btn_select = tk.Button(root, text="Chọn Video & Bắt đầu", command=open_file)
btn_select.pack(pady=20)

root.mainloop()