import os
import subprocess
import ffmpeg
import cv2

input_folder = 'E:\\datasets\\shanghaitech\\training\\videos'
output_folder = 'E:\\datasets\\shanghaitech_recognized\\train\\frames'

# input_folder = 'E:\\datasets\\shanghaitech\\testing\\videos'
# output_folder = 'E:\\datasets\\shanghaitech_xclip\\test\\frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        frame_folder = os.path.splitext(filename)[0]
        frame_filepath = os.path.join(output_folder, frame_folder)
        if not os.path.exists(frame_filepath):
            os.makedirs(frame_filepath)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(frame_filepath, f'{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f'Converted {filename} to {frame_count} frames.')

print('All videos have been converted to frames.')












