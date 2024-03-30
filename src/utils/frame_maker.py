'''Run the script from the current directory'''

import cv2
import os

def extract_frames(video_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame
    for frame_count in range(total_frames):
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Construct the output file path
        output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")

        # Save the frame as PNG
        cv2.imwrite(output_file, frame)

    # Release the video capture object
    cap.release()

# Input video file path
video_file = "../../assets/original_train_videos/moving_circle_14.mp4"

# Output directory for frames
output_dir = "../../assets/original_frames/moving_circle_14"

# Extract frames from the video
extract_frames(video_file, output_dir)
