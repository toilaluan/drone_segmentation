import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from drone_segmentation.model import UperNet, SegFormer, UperNet_HF
from torchmetrics import JaccardIndex
import glob
from PIL import Image
import cv2
import numpy as np
from clearml import Logger
from tqdm import tqdm
import os
classes = ['background', 'small', 'complex', 'simple']
# classes = ['background', 'building', 'street', 'tree', 'water']
class_colors = [
    (0, 0, 0),        # background (black)
    (0, 0, 255),      # building (red)
    (0, 255, 0),      # street (green)
    (0, 255, 255),    # tree (yellow)
    (255, 0, 0)       # water (blue)
]

class LiTSeg(pl.LightningModule):
    def __init__(self, cfg, total_steps, transforms):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms
        if "segformer" in cfg.model.backbone_name:
            self.model = SegFormer(cfg.model.backbone_name)
        elif "upernet" in cfg.model.backbone_name:
            self.model = UperNet_HF(cfg.model.backbone_name, num_labels=cfg.model.upernet_cfg.num_labels)
        else:
            self.model = UperNet(backbone_name=cfg.model.backbone_name, upernet_cfg=cfg.model.upernet_cfg)
        self.total_steps = total_steps
        self.iou_calc = JaccardIndex('multiclass', average='none', num_classes=cfg.model.upernet_cfg.num_labels)
        # self.iou_calc = JaccardIndex('binary')
        self.save_hyperparameters()
    
    def forward(self, x, use_interpolate=True):
        N, C, H, W = x.shape
        logits = self.model(x)
        if use_interpolate:
            logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
        return logits
    
    def training_step(self, batch, batch_id):
        imgs, masks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_id):
        imgs, masks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        mask_prediction = logits.sigmoid()
        self.iou_calc.update(mask_prediction, masks)
    
    def on_validation_epoch_end(self):
        iou_score = self.iou_calc.compute()
        iou_score = iou_score.tolist()
        for i, t in enumerate(classes):
             self.log(f"iou_{t}", iou_score[i], on_epoch=True, prog_bar=True)
        
        self.iou_calc.reset()
       
        predict_masks, original_imgs, img_paths = self.predict_folder(self.cfg.data.test_folder)
        for i, predict_mask in enumerate(predict_masks):
            img_name = os.path.basename(img_paths[i])
            original_img = original_imgs[i]
            color_mask = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)

            # Map the class labels to colors
            for class_idx in range(len(classes)):
                color_mask[predict_mask == class_idx] = class_colors[class_idx]

            # # Overlay the color mask on the original image
            # alpha = 0.5  # Adjust the transparency level
            # result = cv2.addWeighted(original_img, 1, color_mask, alpha, 0)
            # cv2.imwrite("debug.png", result)
            Logger.current_logger().report_image(
                "Testing Image",
                img_name,
                iteration=self.trainer.current_epoch,
                image=color_mask,
            )
        
    def predict_a_image(self, image_path):
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        W, H, C = img.shape
        IMG_SIZE = self.cfg.data.img_size[0]
        STRIDE = IMG_SIZE
        logits_map = np.zeros((W, H))
        for top_left_y in range(0, H, STRIDE):
            for top_left_x in range(0, W, STRIDE):
                # Calculate the bottom right coordinates of the sub-image
                bottom_right_x = min(top_left_x + IMG_SIZE, W)
                bottom_right_y = min(top_left_y + IMG_SIZE, H)
                # Extract the sub-image
                sub_img = img[top_left_x:bottom_right_x, top_left_y:bottom_right_y]
                sub_img = Image.fromarray(sub_img)
                is_last = sub_img.size != (IMG_SIZE, IMG_SIZE)
                ori_sub_size = (sub_img.size[1], sub_img.size[0])
                if is_last:
                    sub_img = sub_img.resize((IMG_SIZE, IMG_SIZE))
                sub_img = self.transforms(sub_img).unsqueeze(0).to(self.device)
                
                # Process the sub-image and update the logits map
                sub_logits = self(sub_img)
                if is_last:
                    sub_logits = F.interpolate(sub_logits, ori_sub_size, mode="bilinear", align_corners=False)
                sub_logits = sub_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                # sub_logits = sub_logits < 0.5
                logits_map[top_left_x:top_left_x+IMG_SIZE, top_left_y:top_left_y+IMG_SIZE] = sub_logits
        predict_mask = np.array(logits_map, dtype=np.uint8)
        return predict_mask, img
    @torch.inference_mode()
    def predict_folder(self, image_dir):
        img_paths = glob.glob(image_dir + '/*')[:1]
        # img_paths = glob.glob(image_dir + '/*')
        predict_masks = []
        original_imgs = []
        for img_path in tqdm(img_paths):
            predict_mask, img = self.predict_a_image(img_path)
            predict_masks.append(predict_mask)
            original_imgs.append(img)
        return predict_masks, original_imgs, img_paths
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            weight_decay=self.cfg.training.weight_decay,
            lr=float(self.cfg.training.lr),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(self.cfg.training.lr),
                    total_steps=self.total_steps,
                    pct_start=0.1,
                    div_factor=25,
                    final_div_factor=1e4,
                ),
                "interval": "step",
            },
        }
        