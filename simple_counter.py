"""
Module đếm người đơn giản.
Chỉ đếm tổng số người trong khung hình, không cần vẽ polygon.
Hỗ trợ pause/resume bằng phím Space.
"""
import cv2
from ultralytics import YOLO


def start_simple_counting(video_path):
    """
    Chế độ đếm đơn giản: đếm tổng số người trong khung hình.
    
    Controls:
        - Space: Pause/Resume video
        - Q: Thoát
    """
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")

    window_name = "Simple People Counting - Press Q to Quit"
    cv2.namedWindow(window_name)

    last_frame = None
    last_people_count = 0
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

        results = model(frame, stream=True)
        people_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    people_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        last_people_count = people_count

        # Hiển thị tổng số người
        label = f"Total People: {people_count}"
        cv2.rectangle(frame, (5, 5), (300, 45), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Hướng dẫn
        cv2.putText(frame, "SPACE: Pause | Q: Quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # Space → pause
            paused = True
            # Hiển thị trạng thái PAUSED lên frame hiện tại
            cv2.rectangle(frame, (5, 50), (450, 90), (0, 0, 0), -1)
            cv2.putText(frame, "PAUSED - Press SPACE to Resume",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow(window_name, frame)

    # Giữ cửa sổ mở sau khi video kết thúc
    if last_frame is not None:
        label = f"Total People: {last_people_count}"
        cv2.rectangle(last_frame, (5, 5), (300, 45), (0, 0, 0), -1)
        cv2.putText(last_frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
