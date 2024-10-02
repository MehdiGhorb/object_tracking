import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cupy as np
from tqdm import tqdm
from src.pyESNN.pyESNcupy import ESN
import imageio.v3 as iio
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize

import time


"""
Build the model
"""

N_INPUTS = 900  # Number of input dimensions (in this case, grayscale pixel values)
N_OUTPUTS = 2  # Number of output dimensions (x and y coordinates)
N_RESERVOIR = 6000  # Number of reservoir neurons
SPECTRAL_RADIUS = 0.5  # Spectral radius of the reservoir weight matrix
INPUT_SCALING = 0.1  # Scaling of the input weights
NOISE = 0.01  # Noise added to each neuron (regularization)
SPARSITY = 0.3  # Proportion of recurrent weights set to zero
FEEDBACK_SCALING = 1.0  # Scaling of the feedback (teacher forcing) weights
TEACHER_SCALING = 1.0  # Scaling of the target signal
TEACHER_FORCING = True  # Use teacher forcing

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

def read_png_image(file_path, i):

    file_name = os.path.basename(file_path)
    image_array = iio.imread(file_path)
    image_array = resize(image_array, (30, 30), anti_aliasing=True)

    # Remove alpha channel
    #image_array = np.flipud(image_array[:,:,:3])

    number = int(file_name.split("_")[1].split(".")[0])

    df = pd.read_csv(f'assets/preprocessed_bb_coordinates/moving_circle_{i}.csv')
    x = df['X-coordinate']
    y = df['Y-coordinate']

    return image_array, x[number], y[number]

def plot_image(img, bb, save_path):

    _, ax = plt.subplots()
    img = img.get()

    plt.imshow(img)
    x, y = (bb[0]-30)//26.7, (bb[1]-30)//20
    rect = patches.Rectangle((x, y), 5, 5, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.savefig(save_path)
    plt.close() 

# Record the start time
start_time = time.time()

X, y_bb = [], []
img_path = os.path.join(os.getcwd(), f'assets/original_frames/moving_circle_0')
files = glob.glob(os.path.join(img_path, '*.png'))
sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))


predictions = []
for i in range(290, 300):
    image_array, x, y = read_png_image(sorted_files[i], 0)

    X.append(image_array)
    y_bb.append([x, y])

X_test, x, y = read_png_image(sorted_files[311], 0)
X.append(X_test)
y_bb.append([x, y])
X, y_bb = np.array(X), np.array(y_bb)

scaler = MinMaxScaler()
scaled_bb = scaler.fit_transform(y_bb.get())

X_train, X_test = X[:-1], X[-1:]
y_bb_train, y_bb_test = scaled_bb[:-1], scaled_bb[-1:]

print(f'shape of X: {X.shape}, shape of y_bb:{y_bb.shape}')

esn.fit(np.array(X_train[:, :, :, 0]).reshape(len(X_train), -1), np.array(y_bb_train), inspect=True)

# predict the test data
bb_scaled = esn.predict(np.array(X_test[:, :, :, 0]).reshape(len(X_test), -1))

# Inverse min-max scaling
bb_scaled = bb_scaled.get()
bb = scaler.inverse_transform(bb_scaled)
print(bb)

bb_pre = 0
bb[bb_pre][0] = abs(bb[bb_pre][0])
bb[bb_pre][1] = abs(bb[bb_pre][1])

plt.figure()
y_bb_test = scaler.inverse_transform(y_bb_test.reshape(1, -1))
plot_image(X_test[bb_pre], y_bb_test[bb_pre], 'ground_truth_time_dependent.png')

plt.figure()

plot_image(X_test[bb_pre], bb[bb_pre], 'prediction_time_dependent.png')

# Calculate the Mean Squared Error (MSE)
mse_loss = np.mean((np.array(bb) - np.array(y_bb_test)) ** 2)
print("Mean Squared Error:", mse_loss)

# Calculate the Mean Absolute Error (MAE)
mae_loss = np.mean(np.abs(np.array(bb) - np.array(y_bb_test)))
print("Mean Absolute Error:", mae_loss)

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
