import glob
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

SAVE_FOLDER = "/mnt/data/luantranthanh/pyramid_segmentation/dataset/full_class_1024"
IMAGE_FOLDER = "/mnt/data/RasterMask_v11/TrueOrtho"
MASK_FOLDER = "/mnt/data/RasterMask_v11/Mask2"
TRAIN_TXT = "/mnt/data/RasterMask_v11/ImageSet/train.txt"
VAL_TXT = "/mnt/data/RasterMask_v11/ImageSet/val.txt"
shutil.rmtree(SAVE_FOLDER, ignore_errors=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)
with open(TRAIN_TXT, "r") as f:
    TRAIN_LIST = f.readlines()
with open(VAL_TXT, "r") as f:
    VAL_LIST = f.readlines()

TRAIN_IMAGE_IDS = [x.strip()[6:].split(".")[0] for x in TRAIN_LIST]
VAL_IMAGE_IDS = [x.strip()[6:].split(".")[0] for x in VAL_LIST]

SIZE = 1024
STRIDE = 512
AREA_THRESHOLD = 0.05


def check_empty_mask(mask):
    return np.all(mask == 0)


def check_area_mask(mask):
    num_non_zeros = np.count_nonzero(mask)
    W, H = mask.shape
    total_pixels = W * H
    ratio = num_non_zeros / total_pixels
    return (ratio < AREA_THRESHOLD) or (ratio > 1 - AREA_THRESHOLD)


def cut_a_image(id, mode):
    image_path = os.path.join(IMAGE_FOLDER, f"Ortho_{id}.tif")
    mask_path = os.path.join(MASK_FOLDER, f"Mask_{id}.tif")
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    print(np.unique(mask))
    W, H, C = image.shape
    assert image[:, :, 0].shape == mask.shape
    cols = np.math.floor((H - SIZE) / STRIDE + 1)
    rows = np.math.floor((W - SIZE) / STRIDE + 1)
    for col in range(cols):
        for row in range(rows):
            sub_id = f"{col}_{row}"
            sub_image = image[
                row * STRIDE : (row + 1) * STRIDE, col * STRIDE : (col + 1) * STRIDE, :
            ]
            sub_mask = mask[
                row * STRIDE : (row + 1) * STRIDE, col * STRIDE : (col + 1) * STRIDE
            ]
            if check_empty_mask(sub_image > 0):
                continue
            save_image_path = os.path.join(
                SAVE_FOLDER, mode, "images", f"Ortho_{id}_{sub_id}.tif"
            )
            save_mask_path = os.path.join(
                SAVE_FOLDER, mode, "masks", f"Mask_{id}_{sub_id}.tif"
            )

            cv2.imwrite(save_image_path, sub_image)
            cv2.imwrite(save_mask_path, sub_mask)


def cut_a_batch(ids, mode):
    os.makedirs(os.path.join(SAVE_FOLDER, mode), exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, mode, "images"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks"), exist_ok=True)
    for id in tqdm(ids):
        cut_a_image(id, mode)


if __name__ == "__main__":
    cut_a_batch(TRAIN_IMAGE_IDS, "train")
    cut_a_batch(VAL_IMAGE_IDS, "val")
