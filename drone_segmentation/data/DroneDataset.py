from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import cv2
import os
import torch
import albumentations as A
import timm
import torchvision.transforms as T

class DroneDataset(Dataset):
    def __init__(self, root_folder: str, img_size: tuple, is_training: bool):
        self.img_size = img_size
        self.is_training = is_training
        if is_training:
            mode = 'train'
        else:
            mode = 'val'
        self.image_dir = os.path.join(root_folder, mode, "images")
        self.mask_dir = os.path.join(root_folder, mode, "masks")
        print(self.image_dir)
        self.image_paths = glob(self.image_dir + "/*")
        self.augment = A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size[0],
                    width=img_size[1],
                    scale=(0.5, 1),
                    ratio=(0.9, 1.1),
                    always_apply=True,
                ),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]
        )
    def __len__(self):
        return len(self.image_paths) 
    
    def resize(self, img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_name = img_path.split("/")[-1].replace("Ortho", "Mask")
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        
        if self.is_training:
            augmented = self.augment(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = self.resize(img, self.img_size)
            mask = self.resize(mask, self.img_size)
        # mask = mask > 0
        img = Image.fromarray(img)
        
        return img, mask
    

class Collator(object):
    def __init__(self, transform, is_train=False, visualize_dir = 'debug'):
        self.transform = transform
        self.is_train = is_train
        self.visualized = False
        self.visualize_dir = visualize_dir
        os.makedirs('debug', exist_ok=True)
        
    def __call__(self, batch):
        imgs = []
        masks = []
        for img, mask in batch:
            imgs.append(img)
            masks.append(mask)
        if not self.visualized:
            for i, (img, mask) in enumerate(zip(imgs, masks)):
                img.save(os.path.join(self.visualize_dir, f"img_{i}.jpg"))
                cv2.imwrite(os.path.join(self.visualize_dir, f"mask_{i}.png"), mask*255)
            self.visualized = True
        imgs = [self.transform(img) for img in imgs]
        masks = [torch.LongTensor(mask).unsqueeze(0) for mask in masks]
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0).squeeze(1)
        return imgs, masks
    
    
def get_transform(backbone_name):
    try:
        data_config = timm.get_pretrained_cfg(backbone_name).to_dict()
        mean = data_config["mean"]
        std = data_config["std"]
    except:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transform

if __name__ == '__main__':
    ds = DroneDataset('/mnt/data/luantranthanh/seg_building/dataset/building_512_binary', (512,512), True)
    print(ds[0])