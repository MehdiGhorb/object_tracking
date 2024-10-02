import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cupy as np
import cv2
from tqdm import tqdm
from src.pyESNN.pyESNcupy import ESN
import imageio.v3 as iio
from io import BytesIO 
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from src.utils.path import *

N_INPUTS = 10000  # Number of input dimensions (in this case, grayscale pixel values)
N_OUTPUTS = 2  # Number of output dimensions (x and y coordinates)
N_RESERVOIR = 8000  # Number of reservoir neurons
SPECTRAL_RADIUS = 0.89  # Spectral radius of the reservoir weight matrix
INPUT_SCALING = 0.6  # Scaling of the input weights
NOISE = 0.3  # Noise added to each neuron (regularization)
SPARSITY = 0.5  # Proportion of recurrent weights set to zero
FEEDBACK_SCALING = 0.9  # Scaling of the feedback (teacher forcing) weights
TEACHER_SCALING = 0.9  # Scaling of the target signal
TEACHER_FORCING = True  # Use teacher forcing

# video config
WIDTH = 100
HEIGHT = 100
DIM = 3
X_DIM = 3

WIDTH_FACTOR = 800 // WIDTH
HEIGHT_FACTOR = 600 // HEIGHT


def draw_bounding_boxes_from_array(video_path, bounding_boxes, output_video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count < len(bounding_boxes):
            x, y = bounding_boxes[frame_count]
            x, y = int(x), int(y)
            width, height = 30, 30 
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Bounding boxes added and new video saved to: {output_video_path}")

def read_png_image(file_path, i):

    file_name = os.path.basename(file_path)
    image_array = iio.imread(file_path)
    image_array = resize(image_array, (100, 100), anti_aliasing=True)

    # Remove alpha channel
    #image_array = np.flipud(image_array[:,:,:3])

    number = int(file_name.split("_")[1].split(".")[0])

    df = pd.read_csv(f'assets/preprocessed_bb_coordinates/moving_circle_{i}.csv')
    x = df['X-coordinate']
    y = df['Y-coordinate']

    return image_array/255.0, x[number]//WIDTH_FACTOR, y[number]//HEIGHT_FACTOR

def plot_image(img, bb, save_path):

    _, ax = plt.subplots()
    img = img.get()
    #bb = bb.get()

    plt.imshow(img)
    x, y = (bb[0]-30)//8, (bb[1]-30)//6
    rect = patches.Rectangle((x, y), 10, 10, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.savefig(save_path)
    plt.close()

#shapes = {'circle':4, 'rect':3, 'star':4}
shapes = {'circle':15}
X, y_bb = [], []
for j in shapes.keys():
    for i in tqdm(range(shapes[j]), desc='Preprocessing images...'):
        img_path = os.path.join(os.getcwd(), f'assets/original_frames/moving_{j}_{i}')
        files = glob.glob(os.path.join(img_path, '*.png'))
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for filepath in sorted_files:
            image_array, x, y = read_png_image(filepath, i)

            X.append(image_array)
            y_bb.append([x, y])
print('Preprocessing done!\n')

X, y_bb = np.array(X), np.array(y_bb)

# Create MinMaxScaler for bounding box data
scaler = MinMaxScaler()
scaled_bb = scaler.fit_transform(y_bb.get())

print(f'shape of X: {X.shape}, shape of y_bb:{y_bb.shape}')

# data splitting
n_samples = len(X)
train_idx = int(n_samples * 0.95)
X_train, X_test = X[:train_idx,...], X[train_idx:, ...]
y_bb_train, y_bb_test = scaled_bb[:train_idx], scaled_bb[train_idx:]

# display a single sample
#plot_image(X_train[10], y_bb_train[10], 'test_image.png')

"""
Build the model
"""

# Create the Echo State Network (ESN)

esn = ESN(n_inputs=N_INPUTS, 
          n_outputs=N_OUTPUTS, 
          n_reservoir=N_RESERVOIR,
          spectral_radius=SPECTRAL_RADIUS, 
          input_scaling=INPUT_SCALING,
          noise=NOISE,
          sparsity=SPARSITY,
          feedback_scaling=FEEDBACK_SCALING,
          teacher_scaling=TEACHER_SCALING,
          teacher_forcing=TEACHER_FORCING, 
          silent=False)

# fit model to the different target data (image, bb coordinates, class label)
esn.fit(np.array(X_train[:, :, :, 0]).reshape(len(X_train), -1), np.array(y_bb_train), inspect=True)
print('Model fitted!\n')

# Inference on the entire video
video_directory = os.path.join(ORIGINAL_VAL_VIDEOS_DIR, 'moving_circle_val_0.mp4')
output_video_path = os.path.join(PREDICTED_VIDEOS_DIR, 'moving_circle_val_0.mp4')

cap = cv2.VideoCapture(video_directory)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    with BytesIO() as f:
        iio.imwrite(f, frame, format='png')
        f.seek(0)
        image_array = iio.imread(f)

    image_array = resize(image_array, (100, 100), anti_aliasing=True)
    frames.append(image_array/255.0)

# Convert frames to numpy array
frames = np.array(frames)
frames = frames.reshape(frames.shape[0], 100, 100, 3)
frames = frames[:, :, :, 0]

predictions = esn.predict(frames.reshape(len(frames), -1))

# Inverse min-max scaling
predictions  = predictions .get()
bb = scaler.inverse_transform(predictions)

for i in range(len(bb)):
    bb[i][0] = int(abs(bb[i][0]) * WIDTH_FACTOR)
    bb[i][1] = int(abs(bb[i][1]) * HEIGHT_FACTOR)
print(bb)

cap.release()

draw_bounding_boxes_from_array(video_directory, bb, output_video_path)
