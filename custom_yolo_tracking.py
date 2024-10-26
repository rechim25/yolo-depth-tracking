from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/rechim/cargotrack/checkpoints/last.pt') 
    results = model.track(source="/home/rechim/object-tracking/240517_062019_062029.mp4", save=True, tracker="bytetrack_depth.yaml")