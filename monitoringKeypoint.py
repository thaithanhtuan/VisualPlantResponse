import cv2
import numpy as np
import os
import glob
import csv
import random


def load_tracking_points(label_file, img_width, img_height):
    tracking_data = {}
    point_types = {}

    with open(label_file, "r") as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            label = int(data[0])
            x, y = int(data[1] * img_width), int(data[2] * img_height)

            if label not in tracking_data:
                tracking_data[label] = []
            tracking_data[label].append((x, y))

    for label, points in tracking_data.items():
        point_types[label] = "flower" if len(points) == 3 else "leaf"

    return tracking_data, point_types


def calculate_angle(p1, p2, p3):
    # Calculate the sum vector from p3->p1 and p3->p2.
    v1 = np.array(p1) - np.array(p3)
    v2 = np.array(p2) - np.array(p3)
    v_sum = v1 + v2

    # Compute the angle of v_sum relative to the horizontal axis.
    angle = np.arctan2(v_sum[1], v_sum[0])
    angle = np.degrees(angle) % 360  # Range 0-360; 0° right, 180° left.
    return angle, v_sum


def track_keypoints(image_folder, label_file, output_folder, movement_threshold=30.0):
    os.makedirs(output_folder, exist_ok=True)
    output_video = os.path.join(output_folder, "tracking_result_F_L.mp4")
    angle_csv = os.path.join(output_folder, "angles.csv")
    point_csv = os.path.join(output_folder, "points.csv")
    movingspeed_csv = os.path.join(output_folder, "movingspeed.csv")

    # Use the "images" subfolder.
    image_folder = os.path.join(image_folder, "images")
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if len(images) < 2:
        print("Not enough images for tracking.")
        return

    base_frame = cv2.imread(images[0])  # For drawing trajectories later.
    first_frame = base_frame.copy()
    img_height, img_width = first_frame.shape[:2]

    tracking_data, point_types = load_tracking_points(label_file, img_width, img_height)
    labels = list(tracking_data.keys())

    # Initialize trajectories for special interest points and previous special points.
    # For flowers, special interest point is the first point (p1); for leaves, it's the only point.
    trajectories = {label: [] for label in labels}
    prev_special = {}
    for label in labels:
        prev_special[label] = tracking_data[label][0]
        trajectories[label].append(tracking_data[label][0])

    # Initialize optical flow tracking with all points.
    prev_points = [pt for label in labels for pt in tracking_data[label]]
    prev_points = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, 10, (img_width, img_height))

    angle_data = []
    point_tracking_data = []
    speed_data = []  # For moving speed of special points.
    frame_interval = 10.0  # seconds between frames

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for i in range(1, len(images)):
        new_frame = cv2.imread(images[i])
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(first_frame, new_frame, prev_points, None, **lk_params)

        output_frame = new_frame.copy()
        frame_angles = [i * frame_interval]  # Time (s)
        frame_points = [i, i * frame_interval]
        frame_speeds = [i, i * frame_interval]  # Time (s) for speed

        index = 0
        for label in labels:
            if point_types[label] == "flower" and index + 2 < len(status):
                if status[index] and status[index + 1] and status[index + 2]:
                    # For each flower, update each point if movement is within threshold; otherwise, retain previous position.
                    if np.linalg.norm(np.array(next_points[index].ravel()) - np.array(
                            prev_points[index].ravel())) <= movement_threshold:
                        p1 = tuple(next_points[index].ravel().astype(int))
                    else:
                        p1 = tuple(prev_points[index].ravel().astype(int))

                    if np.linalg.norm(np.array(next_points[index + 1].ravel()) - np.array(
                            prev_points[index + 1].ravel())) <= movement_threshold:
                        p2 = tuple(next_points[index + 1].ravel().astype(int))
                    else:
                        p2 = tuple(prev_points[index + 1].ravel().astype(int))

                    if np.linalg.norm(np.array(next_points[index + 2].ravel()) - np.array(
                            prev_points[index + 2].ravel())) <= movement_threshold:
                        p3 = tuple(next_points[index + 2].ravel().astype(int))
                    else:
                        p3 = tuple(prev_points[index + 2].ravel().astype(int))

                    # Compute moving speed for the flower tip (p1) relative to its previous special point.
                    current_special = p1
                    dist = np.linalg.norm(np.array(current_special) - np.array(prev_special[label]))
                    speed = dist / frame_interval  # pixels per second
                    frame_speeds.append(speed)
                    prev_special[label] = current_special

                    trajectories[label].append(current_special)

                    angle, v_sum = calculate_angle(p1, p2, p3)
                    # Scale down the sum vector arrow (0.5x length).
                    sum_vector_endpoint = (p3[0] + int(v_sum[0] * 0.5), p3[1] + int(v_sum[1] * 0.5))

                    cv2.circle(output_frame, p1, 5, (0, 0, 255), -1)  # Petal tip
                    cv2.circle(output_frame, p2, 5, (0, 255, 0), -1)  # Sepal
                    cv2.circle(output_frame, p3, 5, (255, 0, 0), -1)  # Bottom pedicel
                    cv2.line(output_frame, p1, p3, (255, 255, 255), 2)
                    cv2.line(output_frame, p2, p3, (255, 255, 255), 2)
                    cv2.arrowedLine(output_frame, p3, sum_vector_endpoint, (255, 255, 0), 1)

                    if angle is not None:
                        frame_angles.append(angle)
                        cv2.putText(output_frame, f"{angle:.2f}°", p3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:
                        frame_angles.append("N/A")

                    frame_points.extend([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]])
                index += 3
            elif point_types[label] == "leaf":
                if status[index]:
                    if np.linalg.norm(np.array(next_points[index].ravel()) - np.array(
                            prev_points[index].ravel())) <= movement_threshold:
                        p1 = tuple(next_points[index].ravel().astype(int))
                    else:
                        p1 = tuple(prev_points[index].ravel().astype(int))
                    current_special = p1
                    dist = np.linalg.norm(np.array(current_special) - np.array(prev_special[label]))
                    speed = dist / frame_interval
                    frame_speeds.append(speed)
                    prev_special[label] = current_special

                    trajectories[label].append(current_special)
                    cv2.circle(output_frame, p1, 5, (255, 255, 0), -1)
                    cv2.arrowedLine(output_frame, tuple(prev_points[index].ravel().astype(int)), p1, (0, 255, 255), 1)
                    frame_points.extend([p1[0], p1[1]])
                index += 1

        video_writer.write(output_frame)
        angle_data.append(frame_angles)
        point_tracking_data.append(frame_points)
        speed_data.append(frame_speeds)
        first_frame = new_frame
        prev_points = next_points

    video_writer.release()

    # Draw trajectories on the base frame using a unique color for each label.
    traj_frame = cv2.imread(images[0])
    colors = {label: tuple(np.random.randint(0, 256, 3).tolist()) for label in labels}
    for label in labels:
        pts = trajectories[label]
        for j in range(1, len(pts)):
            cv2.line(traj_frame, pts[j - 1], pts[j], colors[label], 2)
        cv2.putText(traj_frame, f"Label {label}", pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[label], 2)
    trajectory_image = os.path.join(output_folder, "trajectories.jpg")
    cv2.imwrite(trajectory_image, traj_frame)

    with open(angle_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Time (s)"] + [f"Flower {label} Angle (deg)" for label in labels if point_types[label] == "flower"])
        writer.writerows(angle_data)

    with open(point_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Frame", "Time (s)"]
        for i in range(len(prev_points)):
            header.append(f"Point {i + 1} (x)")
            header.append(f"Point {i + 1} (y)")
        writer.writerow(header)
        writer.writerows(point_tracking_data)

    with open(movingspeed_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Frame", "Time (s)"] + [f"Label {label} Speed (pixels/sec)" for label in labels]
        writer.writerow(header)
        writer.writerows(speed_data)

    print(
        f"Tracking complete. Video saved as {output_video}, angle statistics saved as {angle_csv},\npoint tracking data saved as {point_csv}, moving speed saved as {movingspeed_csv},\nand trajectories image saved as {trajectory_image}.")


# Example usage
image_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/"
label_file = os.path.join(image_folder, "label_F_L.txt")
output_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output"

# Example usage
image_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/"
label_file = os.path.join(image_folder, "label_F_L.txt")
output_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output"
"""
# Example usage
image_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/"
label_file = os.path.join(image_folder, "label_F_L.txt")
output_folder = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output"
"""
track_keypoints(image_folder, label_file, output_folder)