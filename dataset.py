import os
import torch
import torch.utils.data
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import transforms as T
from transforms import MU, STD

import torchxrayvision as xrv
import torch 
import cv2


# Data augementation
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) # converts a PIL image to a PyTorch Tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotate(5))
        transforms.append(T.RandomTranslate(0.1))
    return T.Compose(transforms)


class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root, 
        image_dir='PNGImages',
        ann_file='annotations.json',
        transforms=None, 
        device="cpu",
        ):
        self.root = root
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.device = device

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        
        # load annotations
        f = open(os.path.join(root, ann_file))
        self.json = json.load(f)
        self._load_annotations()

        # load torchxrayvision segmentation model
        self.segmodel = xrv.baseline_models.chestx_det.PSPNet()
        self.segmodel.to(self.device)

        # parameters for CLAHE - Contrast Limited Adaptive Histogram Equalization
        self.clip_limit = 3.0
        self.tile_grid_size = (8, 8)
    

    def _load_annotations(self):
        # image to id
        self.img_to_id = {}
        for img in self.json['images']:
            file_name = img['file_name'].replace('.dcm', '.png')
            self.img_to_id[file_name] = img['id']

        # id to ann
        self.id_to_ann = {}
        for i, ann in enumerate(self.json['annotations']):
            if ann['image_id'] not in self.id_to_ann:
                self.id_to_ann[ann['image_id']] = []
            self.id_to_ann[ann['image_id']].append(i)


    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, self.image_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # process class and bbox for gt
        ann_ids = self.id_to_ann[self.img_to_id[self.imgs[idx]]]
        num_objs = len(ann_ids)

        boxes, labels = [[0,0,0,0], [0,0,0,0]], [0, 0]
        for i in range(num_objs):
            ann = self.json['annotations'][ann_ids[i]]
            # label
            lab = ann['category_id']
            if lab == 3046 or lab == 3047: # only consider carina and ETT
                if labels[lab-3046] == 1 and ann['assignee'] != 'reviewer1@vinbrain.net': # reviewer 1 is the expert
                    continue
                labels[lab-3046] = 1 # flag
                # bbox
                xmin = ann['bbox'][0]
                xmax = ann['bbox'][0] + ann['bbox'][2]
                ymin = ann['bbox'][1]
                ymax = ann['bbox'][1] + ann['bbox'][3]
                boxes[lab-3046] = [xmin, ymin, xmax, ymax]


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Transformations
        img, target = self.transforms(img, target) # Apply augmentations
        target["trachea_mask"] = self.generate_segmask(img) # Generate segmask
        img = self.adjust_contrast(img) # Adjust contrast
        img = (img - MU) / STD # Normalize
        return img, target
    

    def generate_segmask(self, img):
        segmask = img * 255
        segmask = segmask.permute(1, 2, 0).numpy()
        segmask = xrv.datasets.normalize(segmask, 255) # convert 8-bit image to [-1024, 1024] range
        segmask = segmask.mean(2)[None, ...] # Make single color channel
        segmask = torch.from_numpy(segmask).unsqueeze_(0)
        segmask = segmask.to(self.device)
        segmask = self.segmodel(segmask).detach().cpu().numpy()
        segmask = segmask[0, 12, :, :]
        segmask = torch.from_numpy(segmask)
        return segmask
    

    def adjust_contrast(self, img):
        img = img * 255
        img = img.permute(1, 2, 0).numpy()
        img = img.astype('uint8')
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
        equalized_image = clahe.apply(gray_image) / 255
        equalized_image = torch.from_numpy(equalized_image)
        # duplicate 3 times to get 3 channels
        equalized_image = equalized_image.unsqueeze_(0)
        equalized_image = equalized_image.expand(3, -1, -1)
        return equalized_image

    
    def __len__(self):
        return len(self.imgs)


def view_img(image, bbox):
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image * [STD, STD, STD]
    image = image + [MU, MU, MU]
    image = image * 255
    image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in bbox:
        draw.rectangle(box.tolist(), outline='red')
    return image


if __name__ == "__main__":
    dataset = CXRDataset(
            root='/home/ec2-user/data/MIMIC-1105-512', 
            image_dir='PNGImages',
            ann_file='annotations-512.json',
            transforms=get_transform(train=True),
            )
    print(dataset[0])