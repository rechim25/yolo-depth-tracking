#!/bin/bash

# Define the input folder containing .mp4 videos
input_folder="/home/rechim/cargotrack/dataset/videos/mp4/240517"
# Define the output folder for .webp files
output_folder="/home/rechim/cargotrack/dataset/videos/webp/240517"
# Create output folder if it doesn't exist
mkdir -p "$output_folder"

ffmpeg -i input.mp4 -c:v libwebp -lossless 1 output.webp


# Loop over each .mp4 file in the input folder
for video_file in "$input_folder"/*.mp4; do
    # Extract the base name of the video (without extension)
    base_name=$(basename "$video_file" .mp4)

    # Convert to animated WebP
    ffmpeg -i "$video_file" -c:v libwebp -lossless 1 -loop 0 "$output_folder/${base_name}.webp"
done
