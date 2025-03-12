import cv2
import os

# Initialize global variables
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
bounding_boxes = []
current_label = 1  # Default label

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, bounding_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        # Calculate center, width, and height of the bounding box
        x_min = min(start_point[0], end_point[0])
        y_min = min(start_point[1], end_point[1])
        x_max = max(start_point[0], end_point[0])
        y_max = max(start_point[1], end_point[1])

        # Convert to normalized format
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        # Add bounding box to the list
        bounding_boxes.append((current_label, center_x, center_y, width, height))

# Save bounding boxes to a file
def save_bounding_boxes(output_file, bounding_boxes, image_width, image_height):
    with open(output_file, "w") as f:
        for bbox in bounding_boxes:
            l, cx, cy, w, h = bbox
            # Normalize coordinates (0 to 1 range)
            cx /= image_width
            cy /= image_height
            w /= image_width
            h /= image_height
            f.write(f"{l} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# Main function
def main(image_path, output_file):
    global current_label, bounding_boxes

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    image_copy = image.copy()
    image_height, image_width = image.shape[:2]

    # Create a window and set the mouse callback
    cv2.namedWindow("Bounding Box Tool")
    cv2.setMouseCallback("Bounding Box Tool", draw_rectangle)

    while True:
        # Display the image
        temp_image = image.copy()
        for bbox in bounding_boxes:
            l, cx, cy, w, h = bbox
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            x_max = int(cx + w / 2)
            y_max = int(cy + h / 2)
            cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(temp_image, f"Label {l}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if drawing:
            cv2.rectangle(temp_image, start_point, end_point, (255, 0, 0), 2)

        cv2.imshow("Bounding Box Tool", temp_image)

        # Key bindings
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save bounding boxes to the output file
            save_bounding_boxes(output_file, bounding_boxes, image_width, image_height)
            print(f"Bounding boxes saved to {output_file}.")
        elif key == ord("r"):
            # Reset the bounding boxes
            bounding_boxes = []
            print("Bounding boxes reset.")
        elif key == ord("q"):
            # Quit the tool
            break
        elif ord("1") <= key <= ord("9"):
            # Change the label based on the number key pressed
            current_label = key - ord("0")
            print(f"Label changed to {current_label}.")
    cv2.destroyAllWindows()

# Example usage
image_path = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/images/0000_cropped.jpg"  # Replace with your image path
output_file = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/label_F_L.txt"  # Replace with your desired output file name

image_path = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/images/1223_cropped.jpg"  # Replace with your image path
output_file = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/label_F_L.txt"  # Replace with your desired output file name
"""
image_path = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/images/3266_cropped.jpg"  # Replace with your image path
output_file = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/label_F_L.txt"  # Replace with your desired output file name
"""
main(image_path, output_file)