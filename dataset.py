import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

# Dataset - PennFudan
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Dataset - CXR
class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root, 
        image_dir='PNGImages',
        ann_file='annotations.json',
        target_file='target.json',
        transforms=None, 
        first_time=False
        ):
        self.root = root
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.target_file = target_file
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        if first_time:
            # Load annotations and save target data in json
            f = open(os.path.join(root, ann_file))
            self.json = json.load(f)
            self._prepare_dataset()
        else:
            # Load target data
            f = open(os.path.join(root, target_file))
            self.target = json.load(f)
    

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


    def _prepare_dataset(self):
        # Save target data is saved in a json file. Should be called once since it's slow.
        self._load_annotations()
        self.target = []
        counter = 0

        for idx in range(len(self.imgs)):
            ann_ids = self.id_to_ann[self.img_to_id[self.imgs[idx]]]
            num_objs = len(ann_ids)
            boxes, labels = [], []
            for i in range(num_objs):
                ann = self.json['annotations'][ann_ids[i]]
                # label
                lab = ann['category_id']
                if lab != 3046: # only consider ETT for now, TODO: update to include ETT as well
                    continue
                labels.append(1)
                # bbox
                xmin = ann['bbox'][0]
                xmax = ann['bbox'][0] + ann['bbox'][2]
                ymin = ann['bbox'][1]
                ymax = ann['bbox'][1] + ann['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])

            if len(boxes) == 0: # temp fix for pictures without carina
                counter += 1
                print(counter)
                print(self.imgs[idx])
                boxes = [[0, 0, 1, 1]]
                labels = [1]

            image_id = [i]
            area = (boxes[0][3] - boxes[0][1]) * (boxes[0][2] - boxes[0][0])
            # suppose all instances are not crowd
            iscrowd = [0 for _ in range(len(boxes))]

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            self.target.append(target)
        
        # save target data
        with open(os.path.join(self.root, self.target_file), 'w') as outfile:
            json.dump(self.target, outfile)


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.image_dir, self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        # # process class and bbox for gt
        # ann_ids = self.id_to_ann[self.img_to_id[self.imgs[idx]]]
        # num_objs = len(ann_ids)

        # boxes, labels = [], []
        # for i in range(num_objs):
        #     ann = self.json['annotations'][ann_ids[i]]
        #     # label
        #     lab = ann['category_id']
        #     if lab != 3046: # only consider ETT for now, TODO: update to include carina as well
        #         continue
        #     labels.append(lab)
        #     # bbox
        #     xmin = ann['bbox'][0]
        #     xmax = ann['bbox'][0] + ann['bbox'][2]
        #     ymin = ann['bbox'][1]
        #     ymax = ann['bbox'][1] + ann['bbox'][3]
        #     boxes.append([xmin, ymin, xmax, ymax])

        # if len(boxes) == 0:
        #     boxes = [[0, 0, 0, 0]] # temp fix for pictures without ETT
        #     labels = [3046]
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # # masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        target = self.target[idx]
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(target["area"], dtype=torch.float32).unsqueeze(0)
        target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)

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
            target_file='target_downsized.json',
            first_time=True
            )
    print(dataset[0])