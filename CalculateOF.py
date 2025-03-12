import cv2
import numpy as np
import os
from glob import glob

def load_images_from_folder(folder):
    """
    Load tất cả các khung hình từ thư mục timelapse.
    """
    images = []
    for filename in sorted(glob.glob(os.path.join(folder, "*.jpg"))):  # Đảm bảo khung hình được load theo thứ tự
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

def segment_color(image):
    """
    Phân đoạn lá (xanh) và hoa (đỏ) từ khung hình.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color (increased range)
    lower_green = np.array([15, 10, 10], dtype=np.uint8)  # Lowered lower bound for green hue
    upper_green = np.array([115, 255, 255], dtype=np.uint8)  # Raised upper bound for green hue
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Phạm vi màu đỏ (hoa)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    return mask_green, mask_red

def compute_optical_flow(prev_frame, next_frame, mask):
    """
    Tính optical flow giữa hai khung hình dựa trên mask (lá hoặc hoa).
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Optical flow Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Tách độ lớn và hướng chuyển động
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Áp dụng mask
    magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)

    return magnitude, angle

def analyze_timelapse(image_files):
    """
    Phân tích toàn bộ timelapse, tính optical flow cho lá và hoa.
    """
    results = {
        "green": [],  # Chuyển động của lá
        "red": []     # Chuyển động của hoa
    }

    for i in range(len(image_files) - 1):

        # Lấy hai khung hình liên tiếp
        # frame1, frame2 = images[i], images[i + 1]
        frame1 = cv2.imread(image_files[i])
        frame2 = cv2.imread(image_files[i + 1])

        # Phân đoạn lá và hoa
        mask_green, mask_red = segment_color(frame1)

        # Tính optical flow cho lá
        magnitude_green, angle_green = compute_optical_flow(frame1, frame2, mask_green)
        results["green"].append((magnitude_green, angle_green))

        # Tính optical flow cho hoa
        magnitude_red, angle_red = compute_optical_flow(frame1, frame2, mask_red)
        results["red"].append((magnitude_red, angle_red))
        print(i, ":", image_files[i])
        # print(i, ":", magnitude_green,":", angle_green,":", magnitude_red,":", angle_red)
        # print("-------------------------------------")

    return results

def summarize_results(results):
    """
    Tổng hợp kết quả optical flow để rút ra thông tin.
    """
    summary = {
        "green_movement": [],
        "red_movement": []
    }

    # Tổng hợp độ lớn chuyển động
    for mag_green, _ in results["green"]:
        avg_movement_green = np.mean(mag_green[mag_green > 0])  # Chỉ lấy các điểm có chuyển động
        summary["green_movement"].append(avg_movement_green)

    for mag_red, _ in results["red"]:
        avg_movement_red = np.mean(mag_red[mag_red > 0])
        summary["red_movement"].append(avg_movement_red)

    return summary

# --- Chạy chương trình ---
# Thư mục chứa ảnh timelapse
timelapse_folder = "D:/Jeju/Thai/Dataset/Flower timelapse"  # Thay bằng thư mục chứa ảnh
image_files = sorted(glob(os.path.join(timelapse_folder, "*.jpg")))
# images = load_images_from_folder(timelapse_folder)

# Phân tích optical flow
results = analyze_timelapse(image_files)

# Tổng hợp và hiển thị kết quả
summary = summarize_results(results)
print("Chuyển động trung bình của lá qua từng khung hình:", summary["green_movement"])
print("Chuyển động trung bình của hoa qua từng khung hình:", summary["red_movement"])