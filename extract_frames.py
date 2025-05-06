import os
import cv2

def extract_frames(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(save_dir, f"{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1

    cap.release()

def process_all_videos(input_root, output_root):
    for category in ["normal", "abnormal"]:
        category_path = os.path.join(input_root, category)
        if not os.path.isdir(category_path):
            continue

        for file_name in os.listdir(category_path):
            if not file_name.endswith(".mp4"):
                continue

            video_path = os.path.join(category_path, file_name)
            video_folder_name = file_name  # keep .mp4 suffix as folder name
            output_dir = os.path.join(output_root, category, video_folder_name)

            print(f"Processing {video_path} -> {output_dir}")
            extract_frames(video_path, output_dir)

# === 设置路径 ===
input_dir = "G:\\my data\\ped1"
output_dir = "G:\\data\\SDnormal"

process_all_videos(input_dir, output_dir)







