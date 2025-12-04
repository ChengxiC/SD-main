import os
import cv2


def extract_frames(root_dir, out_dir, frame_interval=1):
    classes = ["abnormal", "normal"]

    for cls in classes:
        input_folder = os.path.join(root_dir, cls)
        if not os.path.isdir(input_folder):
            print(f"[Warn] {input_folder} 不存在，跳过该类别。")
            continue

        for fname in os.listdir(input_folder):
            if not fname.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(input_folder, fname)
            video_name = os.path.splitext(fname)[0]

            # 输出目录
            save_folder = os.path.join(out_dir, cls, video_name)
            os.makedirs(save_folder, exist_ok=True)

            print(f"[Info] videos: {video_path}")
            cap = cv2.VideoCapture(video_path)

            frame_idx = 0
            saved_idx = 0   # 实际保存的帧编号（0,1,2,...）

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每隔 frame_interval 帧保存一张
                if frame_idx % frame_interval == 0:
                    # 文件名：0.jpg, 1.jpg, 2.jpg, ...
                    save_name = f"{saved_idx}.jpg"
                    save_path = os.path.join(save_folder, save_name)
                    cv2.imwrite(save_path, frame)
                    saved_idx += 1

                frame_idx += 1

            cap.release()
            print(f"[Done] {video_name}: 共读取 {frame_idx} 帧，保存 {saved_idx} 帧到 {save_folder}")


if __name__ == "__main__":
    root_dir = r"G:\Raw Videos"
    out_dir = r"G:\frames"

    extract_frames(root_dir, out_dir, frame_interval=1)


