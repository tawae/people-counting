"""
Module vẽ vùng polygon trên video frame.
Cho phép người dùng vẽ các polygon bằng chuột trước khi bắt đầu đếm.
"""
import cv2
import numpy as np
from tkinter import messagebox


# Biến module-level cho mouse callback
_current_points = []
_drawing_regions = []


def _mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện chuột khi vẽ polygon."""
    global _current_points, _drawing_regions

    if event == cv2.EVENT_LBUTTONDOWN:
        _current_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Chuột phải để kết thúc vẽ polygon
        if len(_current_points) > 2:
            _drawing_regions.append({
                'type': 'polygon',
                'points': np.array(_current_points),
                'count_in': 0,
                'count_out': 0,
                'inside_count': 0
            })
            _current_points = []
            print("Đã thêm vùng đa giác.")


def draw_regions_ui(video_path):
    """
    Hiển thị frame đầu tiên của video và cho phép người dùng vẽ polygon.
    
    Returns:
        list: Danh sách các region đã vẽ, hoặc None nếu người dùng hủy.
    """
    global _current_points, _drawing_regions

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Lỗi", "Không thể đọc video!")
        return None

    _drawing_regions = []
    _current_points = []

    window_name = "Ve Vung Chon"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print("HƯỚNG DẪN:")
    print("- Click chuột trái để thêm điểm cho đa giác.")
    print("- Chuột phải để kết thúc vẽ đa giác (cần ít nhất 3 điểm).")
    print("- Nhấn 'c' để xóa hết các vùng đã vẽ.")
    print("- Nhấn 'Enter' hoặc 'Space' để BẮT ĐẦU đếm.")
    print("- Nhấn 'q' để thoát.")

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return None

        temp_img = frame.copy()

        # Vẽ các vùng đã hoàn thành
        for region in _drawing_regions:
            pts = region['points']
            cv2.polylines(temp_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(temp_img, "Polygon", tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Vẽ các điểm đang vẽ dở
        if len(_current_points) > 0:
            cv2.polylines(temp_img, [np.array(_current_points)],
                          isClosed=False, color=(0, 0, 255), thickness=2)
            for pt in _current_points:
                cv2.circle(temp_img, pt, 5, (0, 0, 255), -1)

        cv2.putText(temp_img, "Ve da giac (Chuot phai de ket thuc)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, temp_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            _drawing_regions = []
            _current_points = []
        elif key == 13 or key == 32:  # Enter or Space
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return _drawing_regions
