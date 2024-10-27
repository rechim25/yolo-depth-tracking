import os
from ultralytics import YOLO

YOLO_MODEL_PATH = "checkpoints/best.pt"
DATASET_DIR = "./dataset"
RESULTS_DIR = "results-yolo_bytetrack"

if __name__ == '__main__':
    """
    buffer_size: Defines the number of frames an object is retained in memory without detection. Useful for handling temporary occlusions.
    •    Range: 5 - 30
    •    Guidelines:
    •    Lower values (e.g., 5-10) are ideal for less crowded scenes with shorter occlusions.
    •    Higher values (e.g., 20-30) help with crowded scenes or when objects are occluded longer.
    conf_threshold: Sets the minimum confidence level required for an object to be tracked.
    •    Range: 0.3 - 0.6
    •    Guidelines:
    •    Higher thresholds (e.g., 0.5-0.6) reduce false positives but might miss smaller or low-confidence detections.
    •    Lower thresholds (e.g., 0.3-0.4) are useful for detecting smaller or partially occluded objects but may introduce false positives.
    iou_threshold: Determines the minimum IoU (Intersection over Union) required for object association across frames.
    •    Range: 0.3 - 0.7
    •    Guidelines:
    •    Lower values (e.g., 0.3-0.4) make it easier to associate objects across frames but increase the risk of ID switching.
    •    Higher values (e.g., 0.6-0.7) help avoid ID switching but may lose track if objects move significantly.
    max_age: Defines the number of frames to keep an undetected object in memory.
    •    Range: 10 - 60
    •    Guidelines:
    •    Lower values (e.g., 10-20) reduce object retention time, ideal for scenes with fast-moving objects or minimal occlusions.
    •    Higher values (e.g., 30-60) are useful in crowded or cluttered scenes with occlusions, ensuring longer retention for undetected objects.
    tracker: Specifies the tracking algorithm. Options may include deep_sort, bytetrack, etc.
    •    Options:
    •    "deep_sort": Often more stable, suited for generic object tracking.
    •    "bytetrack": Efficient for real-time tracking, particularly in crowded scenes.
    """
    video_files = []
    for item in os.listdir(DATASET_DIR):
        if item.lower().endswith('.mp4'):
            video_files.append(os.path.abspath(
                os.path.join(DATASET_DIR, item)))
    print(video_files)
    video_files = ["./dataset/240517_113611_113711.mp4"]

    model = YOLO(YOLO_MODEL_PATH)

    # Set custom tracker parameters
    # tracker_config = {
    #     # Choose tracking algorithm (e.g., 'bytetrack', 'deep_sort')
    #     "tracker": "bytetrack",
    #     "buffer_size": 15,       # Number of frames to retain memory for unmatched objects
    #     "conf_threshold": 0.4,   # Minimum confidence for detection
    #     "iou_threshold": 0.5,    # Minimum IoU for association
    #     "max_age": 30,           # Frames to keep an undetected object in memory
    # }

    for video_file in video_files:
        results = model.track(
            source=video_file,
            save=True,
            project=RESULTS_DIR,
            name=os.path.basename(video_file),
            tracker="./yolo_tracking.yaml"
        )
