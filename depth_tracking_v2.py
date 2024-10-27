import argparse
import cv2
import numpy as np
import numpy as np
import os
import matplotlib
import time
import torch
from collections import defaultdict
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

YOLO_CKPT_PATH = "/home/rechim/cargotrack/checkpoints/last.pt"
TRACKER_PATH = "/home/rechim/cargotrack/tracker/bytetrack_depth.yaml"
INPUT_VIDEO_PATH = "/home/rechim/object-tracking/240517_062019_062029.mp4"
DEPTH_THRESHOLD = 60


def run_yolo_tracking_with_depth(args):
    # Load an official or custom model
    model = YOLO(YOLO_CKPT_PATH)

    DEVICE = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(
        f"checkpoints/depth_anything_v2_{args.encoder}.pth", map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    os.makedirs(args.outdir, exist_ok=True)

    # Open the input video file
    raw_video = cv2.VideoCapture(INPUT_VIDEO_PATH)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

    # Create output writer for depth
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    output_width = frame_width * 2 + margin_width
    output_path = os.path.join(args.outdir, os.path.splitext(
        os.path.basename(INPUT_VIDEO_PATH))[0] + '.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *"mp4v"), frame_rate, (output_width, frame_height))

    # Store the track history
    track_history = dict()

    # Loop through the video frames
    try:
        i = 0
        while raw_video.isOpened():
            i += 1
            # Read a frame from the video
            success, frame = raw_video.read()
            if not success:
                continue

            # Run Depth-Anything-V2
            t0 = time.time()
            depth = depth_anything.infer_image(frame, args.input_size)
            print(f"Depth inference time: {time.time() - t0}")
            # Normalize depth
            depth_normalized = (depth - depth.min()) / \
                (depth.max() - depth.min()) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_image = (cmap(depth_normalized)[:, :, :3] *
                           255)[:, :, ::-1].astype(np.uint8)

            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker=TRACKER_PATH)

            if results[0].boxes.id is None:
                split_region = np.ones(
                    (frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat(
                    [frame, split_region, depth_image])
                out.write(combined_frame)
                continue

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x = int(x.item())
                y = int(y.item())
                print(f"id {track_id} has x={x}, y={y}")
                if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
                    # Out of bounds detections
                    print(f"id {track_id} out of bounds!")
                    continue
                if track_id not in track_history:
                    # When new object detected apply depth consistency
                    # Find object with similar depth
                    found_similar = False
                    for id in track_history.keys():
                        d = track_history[id][-1]["depth"]
                        print(
                            f"id {id} has depth {d}, new object's depth is {depth[y, x]}")
                        if abs(depth[y, x] - d) < DEPTH_THRESHOLD:
                            # Should be same object, change to new ID
                            track_history[track_id] = track_history.pop(id)
                            # Add new frame data
                            track_history[track_id].append(
                                {"x": float(x), "y": float(y), "depth": depth[y, x]})
                            found_similar = True
                            print(f"Found similar! id {track_id}={id}")
                            break
                    if not found_similar:
                        track_history[track_id] = [
                            {"x": float(x), "y": float(y), "depth": depth[y, x]}]
                else:
                    track_history[track_id].append(
                        {"x": float(x), "y": float(y), "depth": depth[y, x]})
                # Retain 90 tracks for 90 frames
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                # Draw the tracking lines
                points = np.array([(pt["x"], pt["y"])
                                   for pt in track_history[track_id]], dtype=np.int32).reshape((-1, 1, 2))
                print(f"depth image at x,y: {tuple(depth_image[y, x])}")
                cv2.polylines(annotated_frame, [points], isClosed=False, color=tuple(
                    int(c) for c in depth_image[y, x]), thickness=10)
                cv2.putText(annotated_frame, f"ID: {track_id}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)
            # Display the annotated frame
            # cv2.imshow("YOLO11 Tracking", annotated_frame)
            # # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            # Write original + depth to output
            split_region = np.ones(
                (frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat(
                [annotated_frame, split_region, depth_image])
            out.write(combined_frame)
            if i == 200:
                break
    except Exception as e:
        cv2.imshow("Problem here!", frame)
        print(f"Exception: {e}")

    # Release the video capture object and close the display window
    raw_video.release()
    out.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    args = parser.parse_args()

    run_yolo_tracking_with_depth(args)
