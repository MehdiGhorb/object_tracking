import os
import numpy as np
import cv2
from src.pyESNN.pyESN import ESN
from src.main_helper import draw_bounding_boxes

from src.utils.path import *

def main():

    esn = ESN(n_inputs=30000, 
              n_outputs=2, 
              n_reservoir=1000,
              spectral_radius=0.99, 
              input_scaling=0.1,
              silent=False)

    #csv_directory = PREPROCESSED_BB_COORDINATES_DIR
    video_directory = os.path.join(ORIGINAL_VAL_VIDEOS_DIR, 'moving_circle_3.mp4')

    cap = cv2.VideoCapture(video_directory)
    #cap_2 = cv2.VideoCapture(video_path_2)

    # Extract frames and resize them to a smaller size for simplicity
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (100, 100)))

    # Reshape frames for reservoir computing
    frames = np.array(frames).reshape(len(frames), -1)
    #frames_2 = np.array(frames_2).reshape(len(frames_2), -1)

    # Load the trained model
    model_path = os.path.join(MODELS, 'model_2.pkl')
    esn = esn.load(model_path)

    # Predict bounding box coordinates for the entire video
    predictions = esn.predict(frames, continuation=False)

    cap.release()

    output_video_path = os.path.join(PREDICTED_VIDEOS_DIR, 'moving_circle_val_2.mp4')
    draw_bounding_boxes(video_directory, predictions, output_video_path)

if __name__ == "__main__":
    main()