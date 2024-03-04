#from src.main_helper import *
from src.utils.path import *
import numpy as np
from src.pyESN.pyESN import ESN
import cv2
import os
import pandas as pd
import pickle
from tqdm import tqdm

    
#video_path = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_circle.mp4")
#video_path_2 = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_star.mp4")
#csv_path = os.path.join(BB_COORDINATES_DIR, 'moving_circle.csv')
#csv_path_2 = os.path.join(DATA_DIR, 'pedestrian_video/fourway.csv')
#output_video_path = os.path.join(PREDICTED_VIDEOS_DIR, 'moving_star_predicted.mp4')

def main():

    # Define the Echo State Network (ESN) parameters
    N_INPUTS = 480000  # Number of input dimensions (in this case, grayscale pixel values)
    N_OUTPUTS = 2  # Number of output dimensions (x and y coordinates)
    N_RESERVOIR = 1000  # Number of reservoir neurons
    SPECTRAL_RADIUS = 0.99  # Spectral radius of the reservoir weight matrix
    INPUT_SCALING = 0.1  # Scaling of the input weights

    # Create the Echo State Network (ESN)
    esn = ESN(n_inputs=N_INPUTS, 
              n_outputs=N_OUTPUTS, 
              n_reservoir=N_RESERVOIR,
              spectral_radius=SPECTRAL_RADIUS, 
              input_scaling=INPUT_SCALING,
              silent=False)

    # Define paths to the videos
    video_directory = ORIGINAL_VIDEOS_DIR
    video_files = os.listdir(video_directory)

    # Initialize reservoir states and target outputs
    res_states_all_videos = []
    target_outputs_all_videos = []

    # Read and process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_directory, video_file)
        cap = cv2.VideoCapture(video_path)
        res_states = []
        
        # Read target outputs from corresponding CSV file
        csv = os.path.splitext(video_file)[0] + '.csv'
        csv_path = os.path.join(PREPROCESSED_BB_COORDINATES_DIR, csv)
        target_outputs_df = pd.read_csv(csv_path)
        target_outputs = target_outputs_df[['X-coordinate', 'Y-coordinate']].values

        # Initialize reservoir state
        reservoir_state = np.zeros(esn.n_reservoir)
        
        # Read the video frame by frame
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame if needed (e.g., convert to grayscale, resize, etc.)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Flatten the frame to obtain the input vector
            input_vector = gray_frame.flatten().astype(float)
            input_pattern = input_vector.reshape(-1, 1)

            # Run the input vector through the ESN
            reservoir_state = esn._update(reservoir_state, input_pattern, target_outputs[i])

            # Store the reservoir state
            res_states.append(reservoir_state)

            # Update target output
            i += 1

        # Close the video capture
        cap.release()

        # Append reservoir states and target outputs for this video
        res_states_all_videos.extend(res_states)
        target_outputs_all_videos.extend(target_outputs)

    # Reshape the reservoir states array for training
    X = np.array(res_states_all_videos[:-1])  # Input
    y = np.array(target_outputs_all_videos[1:])  # Target output

    # Train the ESN
    esn.fit(X, y)

    # Save the trained model
    model_save_path = os.path.join(MODELS, 'model_0.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(esn, f)



if __name__ == "__main__":
    #predictions = train_and_predict(video_path, video_path_2, csv_path)
    #draw_bounding_boxes(video_path_2, predictions, output_video_path)
    main()