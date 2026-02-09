import cv2
import numpy as np
from ultralytics import YOLO
from sort import * # Thư viện SORT để tracking ID
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# Khởi tạo các biến toàn cục
in_count = 0
out_count = 0
cap = None
model = None
tracker = None
stop_tracking = False

# Biến cho phần vẽ
drawing_regions = []
current_points = []
temp_img = None

def mouse_callback(event, x, y, flags, param):
    global current_points, temp_img, drawing_regions

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Chuột phải để kết thúc vẽ polygon
        if len(current_points) > 2:
            drawing_regions.append({
                'type': 'polygon',
                'points': np.array(current_points),
                'count_in': 0,  # Vào vùng
                'count_out': 0,  # Ra vùng
                'inside_count': 0  # Số người đang ở trong
            })
            current_points = []
            print("Đã thêm vùng đa giác.")

def draw_regions_ui(video_path):
    global temp_img, drawing_regions, current_points
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        messagebox.showerror("Lỗi", "Không thể đọc video!")
        return None

    drawing_regions = []
    current_points = []
    
    cv2.namedWindow("Ve Vung Chon")
    cv2.setMouseCallback("Ve Vung Chon", mouse_callback)
    
    print("HƯỚNG DẪN:")
    print("- Click chuột trái để thêm điểm cho đa giác.")
    print("- Chuột phải để kết thúc vẽ đa giác (cần ít nhất 3 điểm).")
    print("- Nhấn 'c' để xóa hết các vùng đã vẽ.")
    print("- Nhấn 'Enter' hoặc 'Space' để BẮT ĐẦU đếm.")
    print("- Nhấn 'q' để thoát.")

    while True:
        # Kiểm tra nếu cửa sổ bị đóng bằng nút X
        if cv2.getWindowProperty("Ve Vung Chon", cv2.WND_PROP_VISIBLE) < 1:
            return None
            
        temp_img = frame.copy()
        
        # Vẽ các vùng đã hoàn thành
        for region in drawing_regions:
            pts = region['points']
            cv2.polylines(temp_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(temp_img, "Polygon", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Vẽ các điểm đang vẽ dở
        if len(current_points) > 0:
            cv2.polylines(temp_img, [np.array(current_points)], isClosed=False, color=(0, 0, 255), thickness=2)
            for pt in current_points:
                cv2.circle(temp_img, pt, 5, (0, 0, 255), -1)

        cv2.putText(temp_img, "Ve da giac (Chuot phai de ket thuc)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Ve Vung Chon", temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            drawing_regions = []
            current_points = []
        elif key == 13 or key == 32:  # Enter or Space
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()
    return drawing_regions

def start_tracking_process(video_path, regions):
    global in_count, out_count, cap, model, tracker, stop_tracking
    
    if not regions:
        messagebox.showwarning("Cảnh báo", "Bạn chưa vẽ vùng nào!")
        return

    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    track_positions = {} 
    # track_positions[obj_id] = {
    #    'last_pos': (x, y),
    #    'region_states': { region_index: state } 
    # }
    # State for polygon: inside/outside (bool or 1/-1)

    window_name = "People Counting System - Press Q to Quit"
    cv2.namedWindow(window_name)

    last_frame = None

    while cap.isOpened() and not stop_tracking:
        # Kiểm tra nếu cửa sổ bị đóng bằng nút X
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        last_frame = frame.copy()

        results = model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        tracked_objects = tracker.update(detections)
        
        # Reset inside count cho mỗi frame
        for region in regions:
            region['inside_count'] = 0

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Khởi tạo trạng thái nếu là ID mới
            if obj_id not in track_positions:
                track_positions[obj_id] = {'last_pos': midpoint, 'region_states': {}}

            # Kiểm tra từng region
            for i, region in enumerate(regions):
                # --- XỬ LÝ ĐA GIÁC (POLYGON) ---
                # pointPolygonTest returns: +1 (inside), -1 (outside), 0 (on edge)
                is_inside = cv2.pointPolygonTest(region['points'], midpoint, False) 
                
                # Đếm số người đang ở trong polygon (real-time)
                if is_inside >= 0:
                    region['inside_count'] += 1
                
                if i not in track_positions[obj_id]['region_states']:
                    track_positions[obj_id]['region_states'][i] = is_inside
                else:
                    prev_status = track_positions[obj_id]['region_states'][i]
                    
                    # Vào vùng: Từ ngoài (-1) -> vào trong (>=0)
                    if prev_status < 0 and is_inside >= 0:
                        region['count_in'] += 1
                    # Ra vùng: Từ trong (>=0) -> ra ngoài (-1)
                    elif prev_status >= 0 and is_inside < 0:
                        region['count_out'] += 1
                         
                    track_positions[obj_id]['region_states'][i] = is_inside

            track_positions[obj_id]['last_pos'] = midpoint

            # Vẽ bbox và ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Vẽ regions và hiển thị count
        y_offset = 30
        for i, region in enumerate(regions):
            # Hiển thị In, Out và Inside (số người đang trong vùng)
            label = f"R{i}: In:{region['count_in']} Out:{region['count_out']} Inside:{region['inside_count']}"
            color = (0, 255, 255)
            
            cv2.polylines(frame, [region['points']], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Thêm hướng dẫn thoát
        cv2.putText(frame, "Press 'Q' to Quit", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    # Giữ cửa sổ mở sau khi video kết thúc - chờ người dùng bấm thoát thủ công
    if not stop_tracking and last_frame is not None:
        # Vẽ thông báo video kết thúc
        overlay = last_frame.copy()
        cv2.rectangle(overlay, (0, 0), (last_frame.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, last_frame, 0.3, 0, last_frame)
        cv2.putText(last_frame, "VIDEO ENDED - Press 'Q' to Quit", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        while True:
            # Kiểm tra nếu cửa sổ bị đóng bằng nút X
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            cv2.imshow(window_name, last_frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    # Kết thúc
    cap.release()
    cv2.destroyAllWindows()

# --- Giao diện Tkinter ---
def open_file_and_start():
    global stop_tracking
    file_path = filedialog.askopenfilename()
    if file_path:
        # Bước 1: Vẽ vùng chọn
        regions = draw_regions_ui(file_path)
        
        # Bước 2: Bắt đầu đếm nếu có vùng
        if regions:
            stop_tracking = False
            start_tracking_process(file_path, regions)

root = tk.Tk()
root.title("Hệ thống đếm người AI")
root.geometry("300x200")

# label_instr = tk.Label(root, text="Chọn video để bắt đầu.")
# label_instr.pack(pady=10)

btn_select = tk.Button(root, text="Chọn Video & Bắt đầu", command=open_file_and_start)
btn_select.pack(pady=20)

root.mainloop()