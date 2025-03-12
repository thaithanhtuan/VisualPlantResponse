import cv2
import numpy as np
import os
from glob import glob

# Set paths
input_dir = "D:/Jeju/Thai/Dataset/Flower timelapse"  # Directory containing images
output_dir = "Out_dir"  # Directory to save processed images
os.makedirs(output_dir, exist_ok=True)

def extract_plant_pot(image):
    """Extract the plant pot from the image with a white A3 background."""
    # Convert to HSV for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for green color (increased range)
    lower_green = np.array([15, 10, 10], dtype=np.uint8)  # Lowered lower bound for green hue
    upper_green = np.array([115, 255, 255], dtype=np.uint8)  # Raised upper bound for green hue

    # Define range for red color (note: red is split across two ranges in HSV)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for green and red
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Combine green and red masks
    combined_mask = cv2.bitwise_or(mask_green, mask_red)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    return result

def calculate_optical_flow(prev_frame, next_frame):
    """Calculate the optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    # Convert flow to RGB for visualization
    h, w = flow.shape[:2]
    flow_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_rgb[..., 0] = angle * 180 / np.pi / 2  # Hue
    flow_rgb[..., 1] = 255  # Saturation
    flow_rgb[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
    flow_rgb = cv2.cvtColor(flow_rgb, cv2.COLOR_HSV2BGR)
    return flow_rgb

def crop_and_transform(image, src_points):
    """
    Crop and transform an image into a rectangle based on the given corner points.

    Parameters:
        image (numpy.ndarray): Input image.
        src_points (list of tuples): Four corner points of the rectangle in the image.

    Returns:
        numpy.ndarray: Transformed rectangular image.
    """
    # Define the destination points (for the desired rectangle)
    width = int(max(
        np.linalg.norm(src_points[1] - src_points[0]),
        np.linalg.norm(src_points[3] - src_points[2])
    ))
    height = int(max(
        np.linalg.norm(src_points[2] - src_points[0]),
        np.linalg.norm(src_points[3] - src_points[1])
    ))

    # Define the destination rectangle points
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perform the perspective warp
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    return transformed_image

def visualize_optical_flow(magnitude, angle):
    """
    Hiển thị optical flow dưới dạng ảnh màu (HSV).
    """
    h, w = magnitude.shape
    flow_hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Hue: góc của vector chuyển động
    flow_hsv[..., 0] = angle * 180 / np.pi / 2
    # Saturation: giá trị cố định
    flow_hsv[..., 1] = 255
    # Value: độ lớn của chuyển động (chuẩn hóa)
    flow_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Chuyển từ HSV sang BGR để hiển thị
    flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)
    return flow_bgr

def main():
    # Define the source points (four corners of the rectangle in the image)
    src_points = np.array([
        [682, 571],  # Top-left
        [1992, 510],  # Top-right
        [720, 1508],  # Bottom-left
        [2031, 1459]  # Bottom-right
    ], dtype=np.float32)
    # Process images in the directory

    image_files = sorted(glob(os.path.join(input_dir, "*.jpg")))
    for i in range(len(image_files) - 1):
        print(image_files[i])
        # Read current and next frames
        current_image = cv2.imread(image_files[i])
        next_image = cv2.imread(image_files[i + 1])
        # Crop and transform the image
        current_image = crop_and_transform(current_image, src_points)
        output_pot_path = os.path.join(output_dir, f"{i:04d}_cropped.jpg")
        cv2.imwrite(output_pot_path, current_image)

        next_image = crop_and_transform(next_image, src_points)
        # Extract plant pot
        extracted_pot = extract_plant_pot(current_image)
        output_pot_path = os.path.join(output_dir, f"{i:04d}_extracted_pot.jpg")
        cv2.imwrite(output_pot_path, extracted_pot)

        # Calculate optical flow
        flow_image = calculate_optical_flow(current_image, next_image)
        output_flow_path = os.path.join(output_dir, f"{i:04d}_optical_flow.jpg")
        cv2.imwrite(output_flow_path, flow_image)

    print("Processing complete. Results saved in", output_dir)

if __name__ == '__main__':
    main()