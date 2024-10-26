#!/bin/bash

# Path to the folder containing videos
input_folder="/home/rechim/cargotrack/dataset/videos/240517"
# Path to the folder to save the frames
output_folder="/home/rechim/cargotrack/dataset/images/240517"

# Create output folder if it doesn't exist
mkdir -p "$output_folder"

# Loop over each video file in the input folder
for video_file in "$input_folder"/*.mp4; do
    # Extract the base name of the video (without extension)
    video_name=$(basename "$video_file" .mp4)
    
    # Run ffmpeg to extract frames
    ffmpeg -i "$video_file" -vf "fps=15" "$output_folder/${video_name}_%04d.png"
done
