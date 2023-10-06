from clearml import Model, Task
from drone_segmentation import LiTSeg
import os
import cv2
import torch

model_id = 'e2376db34a1340d48c0b6240a14d372f'
ckpt_path = Model(model_id).get_local_copy(raise_on_error=True)

model = LiTSeg.load_from_checkpoint(ckpt_path)
model.to("cuda")
model.eval()

folder_image = '/mnt/data/RasterMask_v11/test_images'
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
predict_masks, original_imgs, img_paths = model.predict_folder(folder_image)
for mask, img_path in zip(predict_masks, img_paths):
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, img_name), mask)