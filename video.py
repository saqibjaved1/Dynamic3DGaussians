# Create video from images

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_video(images, output_path, fps=30):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()


# images in the folder should match the pattern: exp_{save_prefix}_{t}_{camera_id}.png

def create_video_from_folder(folder_path, output_path, save_prefix="", camera_id=0, fps=30):
    
    images = []
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.startswith(f"exp_{save_prefix}_") and filename.endswith("_{:05d}.png".format(camera_id)):
            img = Image.open(os.path.join(folder_path, filename))
            images.append(np.array(img))
    create_video(images, output_path, fps)



exp = "exp"
seq = "basketball"
camera_ids = [0, 1, 2, 3]

for camera_id in camera_ids:
    create_video_from_folder(f"./output/{exp}/{seq}/gt", f"./output/{exp}/{seq}/gt_video_{camera_id}.mp4", "", camera_id)

save_prefixes = ["", "no_mask_prune_avg_0.1", "no_mask_prune_any_0.1", "no_mask_prune_all_0.1"]

# for camera_id in camera_ids:
    # create_video_from_folder(f"./output/{exp}/{seq}/renders", f"./output/{exp}/{seq}/render_video_{save_prefix}_{camera_id}.mp4", save_prefix, camera_id)

# Pretrained model
for camera_id in camera_ids:
    create_video_from_folder(f"./output/{exp}/{seq}/renders", f"./output/{exp}/{seq}/render_video_{camera_id}.mp4", "", camera_id)

# Own models
for camera_id in camera_ids:
    for save_prefix in save_prefixes:
        create_video_from_folder(f"./output/{exp}/{seq}/renders", f"./output/{exp}/{seq}/render_video_{save_prefix}_{camera_id}.mp4", save_prefix, camera_id)