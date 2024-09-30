import cv2
import numpy as np
import csv
import os

# Get the current working directory
current_path = os.getcwd()
parts = current_path.split("object_tracking")
MAIN_DIR = parts[0] + "object_tracking"
ASSETS_DIR = os.path.join(MAIN_DIR, 'assets')
ORIGINAL_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'train_videos_shapes')
BB_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'bb_videos')
BB_COORDINATES_DIR = os.path.join(ASSETS_DIR, 'bb_coordinates')


INPUT_VIDEO_PATH = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_circle.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BB_VIDEOS_DIR, "moving_circle.mp4")
COORDINATESS = os.path.join(BB_COORDINATES_DIR, "moving_circle.csv")

def detect_circle(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply Canny edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Apply HoughCircles to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")

        # Draw circles and bounding boxes
        for (x, y, r) in circles:
            #cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - r - 10, y - r - 10), (x + r + 10, y + r + 10), (0, 255, 0), 2)

        # Extract coordinates of the first detected circle
        x_coord = circles[0][0]
        y_coord = circles[0][1]

        return frame, x_coord, y_coord

    else:
        return frame, '-', '-'
    
def detect_star(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on their shape (star-like)
    stars = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 10:
            stars.append(approx)

    # Draw rectangles around detected stars on the frame
    frame_with_stars = frame.copy()
    for star in stars:
        x, y, w, h = cv2.boundingRect(star)
        cv2.rectangle(frame_with_stars, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract coordinates of the first detected star
    if stars:
        star = stars[0]
        M = cv2.moments(star)
        if M["m00"] != 0:
            x_coord = int(M["m10"] / M["m00"])
            y_coord = int(M["m01"] / M["m00"])
            return frame_with_stars, x_coord, y_coord

    return frame_with_stars, '-', '-'

def detect_rectangle(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on their shape (rectangle)
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles.append(approx)

    # Draw rectangles around detected rectangles on the frame
    frame_with_rectangles = frame.copy()
    for rectangle in rectangles:
        x, y, w, h = cv2.boundingRect(rectangle)
        x -= 10  # Adjust bounding box size
        y -= 10
        w += 20
        h += 20
        cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract coordinates of the first detected rectangle
    if rectangles:
        rectangle = rectangles[0]
        M = cv2.moments(rectangle)
        if M["m00"] != 0:
            x_coord = int(M["m10"] / M["m00"])
            y_coord = int(M["m01"] / M["m00"])
            return frame_with_rectangles, x_coord, y_coord

    return frame_with_rectangles, '-', '-'

def main(input_video_path, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(input_video_path)

    # Get video details
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize CSV writer
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'X-coordinate', 'Y-coordinate'])

        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect circle in the frame
            processed_frame, x_coord, y_coord = detect_circle(frame)

            # Write coordinates to CSV
            if x_coord is not None and y_coord is not None:
                writer.writerow([frame_count, x_coord, y_coord])

            # Write frame with bounding box to output video
            out.write(processed_frame)

            frame_count += 1

            cv2.imshow('Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = INPUT_VIDEO_PATH
    output_video_path = OUTPUT_VIDEO_PATH
    output_csv_path = COORDINATESS
    main(input_video_path, output_video_path, output_csv_path)
