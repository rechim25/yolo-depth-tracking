from ultralytics import YOLO
import time
import os
import numpy as np
from collections import defaultdict
import cv2


class EnhancedBinTracker:
    def __init__(self, memory_duration=5):
        self.tracked_bins = {}
        self.memory_duration = memory_duration
        self.last_positions = {}
        self.current_frame_time = None
        self.next_available_id = 1
        self.active_tracks = set()
        self.current_frame_ids = set()
        # Istoricul pozițiilor pentru fiecare tomberon
        self.bin_history = defaultdict(list)
        self.persistent_tracks = {}  # Track-uri cu istoric consistent

    def calculate_box_center(self, box):
        """Calculează centrul unui box."""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def calculate_distance(self, center1, center2):
        """Calculează distanța euclidiană între două centre."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def calculate_box_area(self, box):
        """Calculează aria box-ului."""
        return (box[2] - box[0]) * (box[3] - box[1])

    def calculate_box_depth(self, box):
        """Estimează adâncimea relativă bazată pe poziția și dimensiunea box-ului."""
        area = self.calculate_box_area(box)
        center_y = (box[1] + box[3]) / 2  # pozitia verticală a centrului
        # Combinăm aria și poziția verticală pentru a estima adâncimea
        # factor de scalare pentru y
        return np.sqrt(area) * (1 + center_y / 1000)

    def get_next_available_id(self):
        """Generează următorul ID disponibil care nu e folosit în frame-ul curent."""
        while self.next_available_id in self.active_tracks or self.next_available_id in self.current_frame_ids:
            self.next_available_id += 1
        return self.next_available_id

    def reset_frame(self):
        """Resetează ID-urile pentru frame-ul curent."""
        self.current_frame_ids.clear()

    def is_persistent_track(self, track_id):
        """Verifică dacă un track este considerat persistent."""
        if track_id in self.persistent_tracks:
            track_data = self.persistent_tracks[track_id]
            consecutive_frames = track_data.get('consecutive_frames', 0)
            # Considerăm track-ul persistent după 5 frame-uri consecutive
            return consecutive_frames >= 5
        return False

    def update_tracking(self, box, frame_idx):
        """Actualizează tracking-ul pentru un tomberon."""
        current_box = box.xyxy[0].tolist()
        current_center = self.calculate_box_center(current_box)
        current_depth = self.calculate_box_depth(current_box)

        best_match = None
        min_score = float('inf')

        # Verificăm track-urile active și recente
        for bin_id, track_data in self.tracked_bins.items():
            if bin_id in self.current_frame_ids:
                continue

            if self.current_frame_time - track_data['last_update'] <= self.memory_duration:
                last_center = self.calculate_box_center(track_data['last_box'])
                last_depth = self.calculate_box_depth(track_data['last_box'])

                # Calculăm distanța în plan
                distance = self.calculate_distance(current_center, last_center)

                # Calculăm diferența de adâncime
                depth_diff = abs(current_depth - last_depth)

                # Calculăm un scor combinat (distanță și adâncime)
                box_diagonal = np.sqrt(
                    (current_box[2] - current_box[0])**2 +
                    (current_box[3] - current_box[1])**2
                )

                # Scor bazat pe mai mulți factori
                distance_score = distance / box_diagonal
                depth_score = depth_diff / current_depth
                movement_score = 1.0

                if 'velocity' in track_data:
                    expected_center = [
                        last_center[0] + track_data['velocity'][0],
                        last_center[1] + track_data['velocity'][1]
                    ]
                    movement_score = self.calculate_distance(
                        current_center, expected_center) / box_diagonal

                total_score = distance_score + depth_score + movement_score

                # Dacă track-ul este persistent, favorizăm menținerea lui
                if self.is_persistent_track(bin_id):
                    total_score *= 0.8

                if total_score < min_score:
                    min_score = total_score
                    best_match = bin_id

        # Pragul pentru potrivire
        score_threshold = 2.0

        if best_match is not None and min_score < score_threshold:
            # Actualizăm track-ul existent
            last_center = self.calculate_box_center(
                self.tracked_bins[best_match]['last_box'])
            velocity = [
                current_center[0] - last_center[0],
                current_center[1] - last_center[1]
            ]

            self.tracked_bins[best_match].update({
                'last_update': self.current_frame_time,
                'last_box': current_box,
                'end_frame': frame_idx,
                'current_box': current_box,
                'velocity': velocity,
                'depth': current_depth
            })

            # Actualizăm istoricul și persistent_tracks
            self.bin_history[best_match].append(current_box)
            if best_match in self.persistent_tracks:
                self.persistent_tracks[best_match]['consecutive_frames'] += 1
            else:
                self.persistent_tracks[best_match] = {'consecutive_frames': 1}

            stable_id = self.tracked_bins[best_match]['stable_id']
            self.current_frame_ids.add(stable_id)
        else:
            # Creăm un nou track
            new_id = self.get_next_available_id()
            self.active_tracks.add(new_id)
            self.current_frame_ids.add(new_id)

            self.tracked_bins[new_id] = {
                'stable_id': new_id,
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'last_update': self.current_frame_time,
                'last_box': current_box,
                'current_box': current_box,
                'velocity': [0, 0],
                'depth': current_depth
            }

            self.bin_history[new_id] = [current_box]
            self.persistent_tracks[new_id] = {'consecutive_frames': 1}
            stable_id = new_id

        return stable_id, current_box

    def clean_old_tracks(self):
        """Șterge track-urile inactive."""
        current_time = self.current_frame_time
        to_remove = []

        for bin_id, track_data in self.tracked_bins.items():
            if current_time - track_data['last_update'] > self.memory_duration:
                to_remove.append(bin_id)
                if bin_id in self.active_tracks:
                    self.active_tracks.remove(bin_id)
                if bin_id in self.persistent_tracks:
                    del self.persistent_tracks[bin_id]

        for bin_id in to_remove:
            del self.tracked_bins[bin_id]
            del self.bin_history[bin_id]


def draw_boxes(img, boxes, labels, color=(0, 255, 0)):
    """Desenează box-urile și etichetele pe imagine."""
    img_copy = img.copy()
    for box, label in zip(boxes, labels):
        # Generăm o culoare unică pentru fiecare ID
        color_id = label * 50 % 255
        unique_color = (color_id, 255 - color_id, 150)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), unique_color, 2)

        # Adaugă text cu fundal pentru mai bună vizibilitate
        text = f"ID: {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Desenează fundal pentru text
        cv2.rectangle(img_copy,
                      (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0], y1),
                      unique_color,
                      -1)

        # Desenează textul
        cv2.putText(img_copy,
                    text,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)

    return img_copy


YOLO_MODEL_PATH = "checkpoints/best.pt"
DATASET_DIR = "./dataset"


def main():
    video_files = []
    for item in os.listdir(DATASET_DIR):
        if item.lower().endswith('.mp4'):
            video_files.append(os.path.abspath(
                os.path.join(DATASET_DIR, item)))
    print(video_files)
    video_files = ["dataset/240517_065436_065446.mp4"]

    # Inițializăm modelul și tracker-ul
    model = YOLO(YOLO_MODEL_PATH)

    # Creăm directorul de ieșire
    output_directory = 'results-feri_deea/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    for video_file in video_files:
        tracker = EnhancedBinTracker(memory_duration=4)

        output_dir_for_video = os.path.join(output_directory, os.path.basename(video_file))
        os.makedirs(output_dir_for_video, exist_ok=False)
        
        # Pregătim video writer
        first_frame = None
        video_writer = None

        # Deschidem fișierul de log
        log_file = os.path.join(output_dir_for_video, f"{os.path.basename(video_file)}.txt")


        tracking_data_dict = defaultdict(lambda: {"first_seen": None, "last_seen": None})
        capture = cv2.VideoCapture(video_file)

        with open(log_file, 'w') as log:
            print("Începem procesarea video...")

            # Rulăm modelul pentru detectare și tracking
            results = model.track(
                source=video_file, stream=True)
            start_processing_time = time.time()

            # Procesăm fiecare frame
            for frame_idx, result in enumerate(results):
                tracker.current_frame_time = time.time()
                tracker.clean_old_tracks()
                tracker.reset_frame()

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                frame_timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                minutes = int(frame_timestamp_ms // 60000)
                seconds = (frame_timestamp_ms % 60000) / 1000
                timestamp_str = f"{minutes:02}:{seconds:05.2f}"

                if result.orig_img is None:
                    print(
                        f"Avertisment: Frame-ul {frame_idx} nu are imagine originală")
                    continue

                img = result.orig_img

                if first_frame is None:
                    first_frame = img
                    output_video = os.path.join(output_dir_for_video, os.path.basename(video_file))
                    height, width = img.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30,
                        (width, height)
                    )

                current_boxes = []
                current_labels = []

                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Sortăm box-urile după adâncime pentru consistență
                    boxes = sorted(
                        result.boxes,
                        key=lambda x: tracker.calculate_box_depth(
                            x.xyxy[0].tolist()),
                        reverse=True
                    )

                    for box in boxes:
                        if box.id is None or box.conf is None:
                            continue

                        stable_id, current_box = tracker.update_tracking(
                            box, frame_idx)

                        current_boxes.append(current_box)
                        current_labels.append(stable_id)

                        if tracking_data_dict[stable_id]["first_seen"] is None:
                            tracking_data_dict[stable_id]["first_seen"] = timestamp_str
                        tracking_data_dict[stable_id]["last_seen"] = timestamp_str

                        log_message = f"Frame {frame_idx}: Stable ID {stable_id}"
                        print(log_message)
                        log.write(log_message + '\n')

                if current_boxes:
                    annotated_frame = draw_boxes(
                        img, current_boxes, current_labels)
                else:
                    annotated_frame = img.copy()

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                cv2.imwrite(os.path.join(output_dir_for_video, "frames", f'frame_{frame_idx}.jpg'), annotated_frame)

                if frame_idx % 100 == 0:
                    print(f"Procesat frame {frame_idx}")

            if video_writer is not None:
                video_writer.release()

            tracking_data = [
                ("Dumpster 1100 L", stable_id, data["first_seen"], True)
                for stable_id, data in tracking_data_dict.items()
            ] + [
                ("Dumpster 1100 L", stable_id, data["last_seen"], False)
                for stable_id, data in tracking_data_dict.items()
            ]
            save_tracking_data_to_excel(tracking_data)

            processing_time = time.time() - start_processing_time
            print("\nStatistici de procesare:")
            print(f"Timp total de procesare: {processing_time:.2f} secunde")
            print(f"Număr total de frame-uri: {frame_idx + 1}")
            print(
                f"Timp mediu per frame: {(processing_time / (frame_idx + 1)):.3f} secunde")
            print("\nProcesarea videoclipului s-a finalizat cu succes.")
            print(f"Video rezultat salvat în: {output_video}")
            print(f"Rezultatele au fost salvate în: {output_dir_for_video}")
            print(f"Log-ul de tracking a fost salvat în: {log_file}")
        capture.release()

import pandas as pd
from datetime import datetime

def save_tracking_data_to_excel(tracking_data, excel_filename="tracked_objects.xlsx"):
    # Initialize a dictionary to store data for each tracked instance
    track_info = []
    object_presence = {}  # Temporary store to keep start time for each track ID
    for obj_type, track_id, timestamp, is_present in tracking_data:
        if is_present:
            # Record start time if object appears
            object_presence[track_id] = timestamp
        else:
            # Record end time if object disappears
            if track_id in object_presence:
                start_time = object_presence.pop(track_id)
                track_info.append({
                    "Object Type": obj_type,
                    "Track ID": track_id,
                    "Start Time": start_time,
                    "End Time": timestamp
                })
    # Convert tracking information to a DataFrame
    df = pd.DataFrame(track_info)
    # Save to Excel file
    df.to_excel(excel_filename, index=False)
    print(f"Tracking data has been saved to {excel_filename}")

if __name__ == "__main__":
    main()
