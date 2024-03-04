import numpy as np
from pyESN import ESN
import cv2

def main():
    # Load the trained model
    model_path = 'path_to_saved_model'
    esn = ESN.load(model_path)

    # Define the video capture
    video_path = 'path_to_unseen_video_file.mp4'
    cap = cv2.VideoCapture(video_path)

    # Initialize reservoir states
    res_states = []

    # Read the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame if needed (e.g., convert to grayscale, resize, etc.)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Flatten the frame to obtain the input vector
        input_vector = gray_frame.flatten().astype(float)

        # Run the input vector through the ESN
        reservoir_state = esn._update(input_vector)

        # Store the reservoir state
        res_states.append(reservoir_state)

    # Reshape the reservoir states array for inference
    X_inference = np.array(res_states)

    # Perform inference to predict the next state
    predicted_states = esn.predict(X_inference)

    # Now, 'predicted_states' contains the predicted reservoir states.
    # You can use these predicted states to derive the x and y coordinates of the object.
    print(predicted_states)

    # Finally, release the video capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()