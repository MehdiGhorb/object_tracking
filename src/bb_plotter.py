import cv2
import pandas as pd

def draw_bounding_boxes(video_path, csv_path, output_video_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for AVI format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read each frame, draw bounding boxes, and write to output video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Extract coordinates for the current frame
        if frame_count < len(df):
            x, y, w, h = df.iloc[frame_count]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        frame_count += 1

    # Release video capture and writer objects
    cap.release()
    out.release()

    print(f"Bounding boxes added and new video saved to: {output_video_path}")

# Example usage
#video_path = 'assets/original_videos/dot_video.mp4'
#csv_path = 'assets/bb_coordinates/bounding_box_coordinates.txt'
#output_video_path = 'assets/bb_videos/dot_bb_video.mp4'

#draw_bounding_boxes(video_path, csv_path, output_video_path)
