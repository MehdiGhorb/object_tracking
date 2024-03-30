import os

# Get the current working directory
current_path = os.getcwd()
parts = current_path.split("object_tracking")
MAIN_DIR = parts[0] + "object_tracking"

# Define the data directory
DATA_DIR = os.path.join(MAIN_DIR, 'data')
ASSETS_DIR = os.path.join(MAIN_DIR, 'assets')

ORIGINAL_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'original_train_videos')
ORIGINAL_VAL_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'original_val_videos')
BB_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'bb_videos')
BB_COORDINATES_DIR = os.path.join(ASSETS_DIR, 'bb_coordinates')
PREDICTED_VIDEOS_DIR = os.path.join(ASSETS_DIR, 'predictions')
PREPROCESSED_BB_COORDINATES_DIR = os.path.join(ASSETS_DIR, 'preprocessed_bb_coordinates')

MODELS = os.path.join(MAIN_DIR, 'models')

VISUALS = os.path.join(MAIN_DIR, 'visuals')