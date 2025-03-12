import cv2
import numpy as np
import os
from glob import glob

def segment_red_flower(image):
    """
    Phát hiện các hoa màu đỏ trong ảnh.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Phạm vi màu đỏ trong không gian HSV
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Tạo mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    return mask
def detect_flowers(mask):
    """
    Xác định các hoa và hướng của từng hoa từ mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flowers = []

    for contour in contours:
        # Bỏ qua các contour quá nhỏ hoặc quá lớn (giới hạn diện tích)
        area = cv2.contourArea(contour)
        if area < 1000 or area > 20000:  # Điều chỉnh ngưỡng theo kích thước hoa
            continue

        # Tính PCA hoặc FitEllipse để tìm hướng
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center = ellipse[0]  # Tâm của ellipse
            angle = ellipse[2]  # Góc nghiêng của ellipse
            flowers.append({"center": center, "angle": angle, "contour": contour})
        else:
            # Sử dụng PCA để xác định hướng (trong trường hợp không thể fitEllipse)
            # contour_points = np.reshape(contour, (-1, 2))
            # Chuyển đổi các điểm contour thành float32
            contour_points = np.reshape(contour, (-1, 2)).astype(np.float32)
            mean_eigenvectors = cv2.PCACompute2(contour_points, mean=None)
            mean = mean_eigenvectors[0]  # Lấy mean
            eigenvectors = mean_eigenvectors[1]  # Lấy eigenvectors
            center = mean[0]
            principal_axis = eigenvectors[0]
            angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi
            flowers.append({"center": center, "angle": angle, "contour": contour})

    return flowers
def match_flowers(prev_frame, next_frame, prev_flowers, next_flowers):
    """
    Ghép cặp hoa giữa hai frame liên tiếp bằng Optical Flow.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Lấy tâm của các hoa trong frame trước
    prev_points = np.array([flower["center"] for flower in prev_flowers], dtype=np.float32).reshape(-1, 1, 2)

    # Tính Optical Flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None)

    # Ghép cặp hoa
    matched_flowers = []
    for i, (prev_flower, next_point) in enumerate(zip(prev_flowers, next_points)):
        if status[i]:  # Kiểm tra nếu Optical Flow tìm thấy điểm tương ứng
            next_point = tuple(next_point[0])
            for next_flower in next_flowers:
                if np.linalg.norm(np.array(next_point) - np.array(next_flower["center"])) < 20:  # Ngưỡng khoảng cách
                    matched_flowers.append({
                        "prev_flower": prev_flower,
                        "next_flower": next_flower
                    })
                    break
    return matched_flowers
def compute_flower_motion(matched_flowers, time_interval):
    """
    Tính tốc độ rủ hoặc hồi phục của từng hoa.
    """
    results = []
    for pair in matched_flowers:
        prev_angle = pair["prev_flower"]["angle"]
        next_angle = pair["next_flower"]["angle"]

        angle_change = next_angle - prev_angle  # Sự thay đổi góc
        speed = angle_change / time_interval  # Tốc độ thay đổi góc
        results.append({
            "prev_flower": pair["prev_flower"],
            "next_flower": pair["next_flower"],
            "angle_change": angle_change,
            "speed": speed
        })
    return results
def analyze_timelapse(image_files, time_interval=10):
    """
    Phân tích chuỗi ảnh timelapse để theo dõi chuyển động của hoa.
    """
    results = []
    for i in range(len(image_files) - 1):
        # Lấy hai frame liên tiếp
        print(i," :", image_files[i])
        prev_frame = cv2.imread(image_files[i])
        next_frame = cv2.imread(image_files[i + 1])

        # Phát hiện hoa
        prev_mask = segment_red_flower(prev_frame)
        next_mask = segment_red_flower(next_frame)

        prev_flowers = detect_flowers(prev_mask)
        next_flowers = detect_flowers(next_mask)

        # Ghép cặp hoa
        matched_flowers = match_flowers(prev_frame, next_frame, prev_flowers, next_flowers)

        # Tính tốc độ rủ hoặc hồi phục
        motion_results = compute_flower_motion(matched_flowers, time_interval)
        results.append(motion_results)

        print(f"Frame {i} -> Frame {i+1}: {len(motion_results)} hoa được theo dõi.")
    return results



# Thư mục chứa ảnh timelapse

timelapse_folder = "D:/Jeju/Thai/Dataset/Flower timelapse"  # Thay bằng thư mục chứa ảnh
image_files = sorted(glob(os.path.join(timelapse_folder, "*.jpg")))

# Phân tích chuỗi ảnh
time_interval = 10  # Khoảng cách thời gian giữa các khung hình (giây)
results = analyze_timelapse(image_files, time_interval)

# In kết quả
for frame_result in results:
    for motion in frame_result:
        print(f"Hoa: {motion['prev_flower']['center']} -> {motion['next_flower']['center']}")
        print(f"Góc thay đổi: {motion['angle_change']:.2f} độ, Tốc độ: {motion['speed']:.2f} độ/giây")
