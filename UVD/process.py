import os
import re
import cv2
import pandas as pd
import subprocess

def sort_key_func(file):
    numbers = re.findall(r'\d+', file)
    return int(numbers[0]) if numbers else float('inf')

def images_to_video(image_folder, output_video, fps=30, img_extension='.png'):
    try:
        images = [img for img in os.listdir(image_folder) if img.endswith(img_extension)]
        images.sort(key=sort_key_func)
        if not images:
            print("No images found in the directory.")
            return

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        if frame is None:
            print("Error reading the first image.")
            return
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        video.release()
    except Exception as e:
        print(f"Error processing video: {e}")

def run_script_and_get_index(video_path):
    try:
        python_exec_path = ''
        ru1_script_path = ''
        command = f"{python_exec_path} {ru1_script_path} '{video_path}'"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()

        print("Full output")
        print(output)

        for line in output.splitlines():
            if "Subgoal indices:" in line:
                index_str = line.split("Subgoal indices:")[-1].strip()
                return index_str

        raise ValueError("Output does not contain expected prefix 'Subgoal indices:'")
    except Exception as e:
        print(f"Error running ru1.py: {e}")
        return "Error"

def process_folders(root_dir):
    data = []
    try:
        for subdir, dirs, files in os.walk(root_dir):
            for dir in dirs:
                if dir in ['bag_images', 'bag_side_images', 'bag_top_images']:
                    rgb_path = os.path.join(subdir, dir, 'rgb')
                    if os.path.exists(rgb_path):
                        video_path = os.path.join(rgb_path, "output_video.mp4")
                        images_to_video(rgb_path, video_path, fps=30)
                        index = run_script_and_get_index(video_path)  # 使用视频路径
                        # 从每个路径中删除根目录部分
                        relative_path = os.path.relpath(rgb_path, root_dir)
                        data.append([relative_path, index])
    except Exception as e:
        print(f"Error during folder processing: {e}")

    return data

# 设定根目录并开始处理所有文件夹
root_directory = '/'
data = process_folders(root_directory)

# 保存到CSV
df = pd.DataFrame(data, columns=['Directory', 'Index'])
df.to_csv('', index=False)

print("Done processing all folders and saved to CSV.")
