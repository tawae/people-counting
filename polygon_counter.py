"""
Module đếm người với polygon (In/Out tracking).
Sử dụng SORT tracker để theo dõi ID và đếm người vào/ra vùng polygon.
Hỗ trợ pause/resume bằng phím Space.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from tkinter import messagebox


def start_tracking_process(video_path, regions):
    """
    Chế độ đếm In/Out với polygon regions và SORT tracker.
    
    Args:
        video_path: Đường dẫn video.
        regions: Danh sách các polygon region từ draw_regions_ui().
    
    Controls:
        - Space: Pause/Resume video
        - Q: Thoát
    """
    if not regions:
        messagebox.showwarning("Cảnh báo", "Bạn chưa vẽ vùng nào!")
        return

    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # track_positions[obj_id] = {
    #    'last_pos': (x, y),
    #    'region_states': { region_index: state }
    # }
    track_positions = {}

    window_name = "People Counting System - Press Q to Quit"
    cv2.namedWindow(window_name)

    last_frame = None
    paused = False

    while cap.isOpened():
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Xử lý pause
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key == 32:  # Space
                paused = False
            elif key == ord('q') or key == ord('Q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()

        # Detect người
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

        # Track objects
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
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Vẽ regions và hiển thị count
        y_offset = 30
        for i, region in enumerate(regions):
            label = f"R{i}: In:{region['count_in']} Out:{region['count_out']} Inside:{region['inside_count']}"
            color = (0, 255, 255)

            cv2.polylines(frame, [region['points']], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # Hướng dẫn
        cv2.putText(frame, "SPACE: Pause | Q: Quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # Space → pause
            paused = True
            # Hiển thị trạng thái PAUSED
            cv2.rectangle(frame, (5, y_offset), (450, y_offset + 35), (0, 0, 0), -1)
            cv2.putText(frame, "PAUSED - Press SPACE to Resume",
                        (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow(window_name, frame)

    # Giữ cửa sổ mở sau khi video kết thúc - hiển thị chỉ số cuối cùng
    if last_frame is not None:
        y_offset = 30
        for i, region in enumerate(regions):
            label = f"R{i}: In:{region['count_in']} Out:{region['count_out']} Inside:{region['inside_count']}"
            color = (0, 255, 255)
            cv2.polylines(last_frame, [region['points']], isClosed=True, color=color, thickness=2)
            cv2.putText(last_frame, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        cv2.putText(last_frame, "VIDEO ENDED - Press 'Q' to Quit",
                    (10, last_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.imshow(window_name, last_frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    cap.release()
    cv2.destroyAllWindows()
