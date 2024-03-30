import cv2
import pandas as pd
from tqdm import trange
#from sklearn.model_selection import train_test_split
from src.pyESNN.pyESN import ESN
import numpy as np


def train_and_predict(video_path, video_path_2, csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    #cap_2 = cv2.VideoCapture(video_path_2)

    # Extract frames and resize them to a smaller size for simplicity
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (100, 100)))
    '''
    # unseen video
    frames_2 = []
    while cap.isOpened():
        ret, frame = cap_2.read()
        if not ret:
            break
        frames_2.append(cv2.resize(frame, (100, 100)))
    '''

    # Extract bounding box coordinates for training
    #y_train = df[['x', 'y', 'w', 'h']].values
    df.drop(columns=['Frame'], inplace=True)
    y_train = df[['X-coordinate', 'Y-coordinate']].values

    # Reshape frames for reservoir computing
    frames = np.array(frames).reshape(len(frames), -1)
    #frames_2 = np.array(frames_2).reshape(len(frames_2), -1)

    # Create Echo State Network (ESN) reservoir
    reservoir_size = 500
    esn = ESN(n_inputs=frames.shape[1], 
              n_outputs=y_train.shape[1], 
              n_reservoir=reservoir_size)

    # Train the ESN with all frames
    esn.fit(frames, y_train)

    # Predict bounding box coordinates for the entire video
    predictions = esn.predict(frames)

    # Release video capture object
    cap.release()
    #cap_2.release()

    return predictions

def draw_bounding_boxes(video_path, predictions, output_video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for AVI format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read each frame, draw bounding boxes based on predictions, and write to output video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Extract predicted coordinates for the current frame
        if frame_count < len(predictions):
            #x, y, w, h = predictions[frame_count]
            #x, y, w, h = int(x), int(y), int(w), int(h)
            x, y = predictions[frame_count]
            try:
                x, y = int(x), int(y)
            except:
                x, y = 0, 0

            # Draw bounding box
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + 30, y + 30), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        frame_count += 1

    # Release video capture and writer objects
    cap.release()
    out.release()

    print(f"Bounding boxes added and new video saved to: {output_video_path}")

def resize_frames(frames, new_width, new_height):
    resized_frames = []

    for frame in frames:
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames.append(resized_frame)

    return resized_frames