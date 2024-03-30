import os
import cv2
import pandas as pd
import cupy as cp
from tqdm import tqdm
from src.pyESNN.pyESNcupy import ESN
from src.utils.path import *
from src.main_helper import draw_bounding_boxes

def main():
    # Define the Echo State Network (ESN) parameters
    N_INPUTS = 30000  # Number of input dimensions (in this case, grayscale pixel values)
    N_OUTPUTS = 2  # Number of output dimensions (x and y coordinates)
    N_RESERVOIR = 700  # Number of reservoir neurons
    SPECTRAL_RADIUS = 0.99  # Spectral radius of the reservoir weight matrix
    INPUT_SCALING = 0.1  # Scaling of the input weights

    # Create the Echo State Network (ESN)
    esn = ESN(n_inputs=N_INPUTS, 
              n_outputs=N_OUTPUTS, 
              n_reservoir=N_RESERVOIR,
              spectral_radius=SPECTRAL_RADIUS, 
              input_scaling=INPUT_SCALING,
              teacher_forcing=True,
              silent=False)

    # Define paths to the videos
    csv_directory = PREPROCESSED_BB_COORDINATES_DIR
    video_directory = ORIGINAL_VIDEOS_DIR
    video_files = os.listdir(video_directory)

    # Read and process each video
    for video_file in tqdm(video_files, desc="Processing videos ..."):
        video_path = os.path.join(video_directory, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Read target outputs from corresponding CSV file
        csv_file = os.path.splitext(video_file)[0] + '.csv'
        csv_path = os.path.join(csv_directory, csv_file)
        target_outputs_df = pd.read_csv(csv_path)
        target_outputs = target_outputs_df[['X-coordinate', 'Y-coordinate']].values.astype(float)
        
        # Read the video frame by frame
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (100, 100)))
            
        frames = cp.array(frames).reshape(len(frames), -1)           

        # Close the video capture
        cap.release()

        #print('FRAME ..............', frames.shape)
        #print('Y TRAIN ....................', target_outputs.shape)

        # Train the ESN
        print('Training the ESN ...')
        print(frames.shape, target_outputs.shape)
        esn.fit(frames, target_outputs, inspect=True)


    # Extract frames and resize them to a smaller size for simplicity
    cap = cv2.VideoCapture(os.path.join(ORIGINAL_VAL_VIDEOS_DIR, 'moving_circle_3.mp4'))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (100, 100)))

    frames = cp.array(frames).reshape(len(frames), -1)
    
    predictions = esn.predict(frames)

    # Draw bounding boxes on the video
    output_video_path = os.path.join(PREDICTED_VIDEOS_DIR, 'moving_circle_val_2.mp4')
    draw_bounding_boxes(video_directory, predictions, output_video_path)

    
    # Save the trained model
    model_path = os.path.join(MODELS, 'model_2.pkl')
    esn.save(model_path)

if __name__ == "__main__":
    main()
