import os
from ultralytics import YOLO


YOLO_MODEL_PATH = "checkpoints/best.pt"
DATASET_DIR = "./dataset"
RESULTS_DIR = "results-radu_with_deea_model"

if __name__ == '__main__':
    video_files = []
    for item in os.listdir(DATASET_DIR):
        if item.lower().endswith('.mp4'):
            video_files.append(os.path.abspath(
                os.path.join(DATASET_DIR, item)))
    print(video_files)

    model = YOLO(YOLO_MODEL_PATH)
    for video_file in video_files:
        results = model.track(
            source=os.path.abspath(video_file),
            save=True,
            tracker="bytetrack_depth.yaml",
            project=RESULTS_DIR,
            name=os.path.basename(video_file),
        )
