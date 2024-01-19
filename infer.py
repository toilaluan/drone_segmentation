from clearml import Model, Task
from drone_segmentation import LiTSeg
import os
import cv2
import torch

# segformer b3: 3d06e266f1884f2b93c62aced8224a2e
# efficientvit b3 - upernet: 22b1f065b84345f687663ba002bed981
# swin tiny - upernet: 2a45243c407f4bbaa906d205164ebe35
# convnext - upernet: e2376db34a1340d48c0b6240a14d372f

model_id = '2a45243c407f4bbaa906d205164ebe35'
ckpt_path = Model(model_id).get_local_copy(raise_on_error=True)

model = LiTSeg.load_from_checkpoint(ckpt_path)
model.to("cuda")
model.eval()

folder_image = '/mnt/data/RasterMask_v11/test_images'
output_dir = "out/test"
os.makedirs(output_dir, exist_ok=True)
predict_masks, original_imgs, img_paths = model.predict_folder(folder_image)
for mask, img_path in zip(predict_masks, img_paths):
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, img_name), mask)