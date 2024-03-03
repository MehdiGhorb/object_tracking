from src.main_helper import *
from src.utils.path import *


    
video_path = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_circle.mp4")
video_path_2 = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_star.mp4")
csv_path = os.path.join(BB_COORDINATES_DIR, 'moving_circle.csv')
#csv_path_2 = os.path.join(DATA_DIR, 'pedestrian_video/fourway.csv')
output_video_path = os.path.join(PREDICTED_VIDEOS_DIR, 'moving_star_predicted.mp4')

if __name__ == "__main__":
    predictions = train_and_predict(video_path, video_path_2, csv_path)
    draw_bounding_boxes(video_path_2, predictions, output_video_path)
