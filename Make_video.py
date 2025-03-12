import cv2
import os
import os.path

def create_timelapse_video(image_folder, output_video, fps=30):
    """
    Combine timelapse images into a video using OpenCV.

    :param image_folder: Path to the folder containing timelapse images
    :param output_video: Output video file path (e.g., "timelapse.mp4")
    :param fps: Frames per second for the video
    """
    # Get the list of image files
    images = sorted(
        [img for img in os.listdir(image_folder) if (img.endswith(".jpg"))]
    )

    if not images:
        print("No images found in the folder.")
        return

    # Get the size of the first image
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # ROTATE_90_COUNTERCLOCKWISE # ROTATE_90_CLOCKWISE
    height, width, _ = frame.shape

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write to the video
    for image in images:

        image_path = os.path.join(image_folder, image)
        print(image_path)
        frame = cv2.imread(image_path)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # ROTATE_90_COUNTERCLOCKWISE # ROTATE_90_CLOCKWISE
        if frame is None:
            print(f"Error reading {image_path}, skipping...")
            continue

        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()
    print(f"Timelapse video saved to {output_video}")


# Example usage
image_folder = "D:/Jeju/Thai/Research/Plant Optical Flow/Code/Out_dir"
output_video = "timelapse_bg.mp4"
fps = 10  # Adjust frame rate as needed

image_folder = "D:/Jeju/Thai/Dataset/Timelapse Corn Buckwheat"
plant = "Buckwheat"
output_video = os.path.join(image_folder, plant + ".mp4")
image_folder = os.path.join(image_folder, plant)
fps = 10  # Adjust frame rate as needed

image_folder = "D:/Jeju/Thai/Dataset/Timelapse Corn Buckwheat"
plant = "Corn 1"
output_video = os.path.join(image_folder, plant + "_rotate.mp4")
image_folder = os.path.join(image_folder, plant)
fps = 10  # Adjust frame rate as needed

image_folder = "D:/Jeju/Thai/Dataset/Timelapse Corn Buckwheat"
plant = "pi1_Corn"
output_video = os.path.join(image_folder, plant + "_rotate.mp4")
image_folder = os.path.join(image_folder, plant)
fps = 10  # Adjust frame rate as needed

create_timelapse_video(image_folder, output_video, fps)