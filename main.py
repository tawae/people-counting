"""
Hệ thống đếm người AI — Entry point.
Giao diện Tkinter cho phép chọn giữa 2 chế độ đếm:
  1. Vẽ vùng polygon & đếm In/Out
  2. Đếm đơn giản (tổng số người trong frame)
"""
import tkinter as tk
from tkinter import filedialog

from region_drawer import draw_regions_ui
from polygon_counter import start_tracking_process
from simple_counter import start_simple_counting


def open_file_and_start():
    """Chế độ vẽ vùng polygon rồi đếm In/Out."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if file_path:
        regions = draw_regions_ui(file_path)
        if regions:
            start_tracking_process(file_path, regions)


def open_file_simple_count():
    """Chế độ đếm đơn giản — không vẽ vùng, chỉ đếm tổng số người."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if file_path:
        start_simple_counting(file_path)


# --- Tkinter UI ---
root = tk.Tk()
root.title("Hệ thống đếm người AI")
root.geometry("350x250")

label_title = tk.Label(root, text="Chọn chế độ đếm người", font=("Helvetica", 14, "bold"))
label_title.pack(pady=15)

btn_polygon = tk.Button(root, text="Vẽ vùng & Đếm In/Out",
                        command=open_file_and_start, width=25)
btn_polygon.pack(pady=10)

btn_simple = tk.Button(root, text="Đếm toàn bộ người trong khung hình",
                       command=open_file_simple_count, width=25)
btn_simple.pack(pady=10)

root.mainloop()