import os
import torch
import torch.utils.data
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import transforms as T
# import torchvision.transforms as TT


# Data augementation
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) # converts a PIL image to a PyTorch Tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root, 
        image_dir='PNGImages',
        ann_file='annotations.json',
        transforms=None, 
        ):
        self.root = root
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        
        # load annotations
        f = open(os.path.join(root, ann_file))
        self.json = json.load(f)
        self._load_annotations()
    

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
            # self.id_to_ann[ann['image_id']].append(ann['id'])
            self.id_to_ann[ann['image_id']].append(i)


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.image_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # process class and bbox for gt
        ann_ids = self.id_to_ann[self.img_to_id[self.imgs[idx]]]
        num_objs = len(ann_ids)

        boxes, labels = [], []
        for i in range(num_objs):
            ann = self.json['annotations'][ann_ids[i]]
            # label
            lab = ann['category_id']
            if lab != 3046: # only consider ETT for now, TODO: update to include carina as well
                continue
            labels.append(1)
            # bbox
            xmin = ann['bbox'][0]
            xmax = ann['bbox'][0] + ann['bbox'][2]
            ymin = ann['bbox'][1]
            ymax = ann['bbox'][1] + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        # if len(boxes) == 0:
        #     boxes = [[0, 0, 0, 0]] # temp fix for pictures without ETT
        #     labels = [1]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = CXRDataset(
            root='/home/ec2-user/data/MIMIC_ETT_annotations', 
            image_dir='downsized',
            ann_file='annotations_downsized.json',
            transforms=get_transform(train=True),
            )
    print(dataset[0])