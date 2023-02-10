import numpy as np
import torch
import torch.utils.data

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import ops

from dataset import *

# Data augementation
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Create an instance segmentation model based on pretrained models
def get_instance_segmentation_model(num_classes):
    # # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)

    # Update transform size dimension to fit CXR extra-large images
    model.transform.min_size = (2000, )
    model.transform.max_size = 3333
    print(model.transform)
    return model


def pipeline(dataset_name='CXRDataset'):
    # Parameters
    num_classes = 2 # our dataset has two classes only - background and person
    num_epochs = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    if dataset_name == 'PennFudanDataset':
        dataset = PennFudanDataset('PennFudanPed', get_transform(train=False)) # temporarily disable data augmentation, TODO: fix this
        dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    else:
        dataset = CXRDataset(
            root='/home/ec2-user/data/MIMIC_ETT_annotations', 
            transforms=get_transform(train=False)
            )
        dataset_test = CXRDataset(
            root='/home/ec2-user/data/MIMIC_ETT_annotations', 
            transforms=get_transform(train=False)
            )

    # split the dataset in train and test set
    torch.manual_seed(1234)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    model.eval()
    model = nn.Sequential(
        model.transform,
        model.backbone
    )
    for images, targets in dataset:
        images = images.to(device)
        images = [images]
        # x = model.transform(images)
        # x = model.backbone(x[0].tensor)
        # print(x)
        predicted = model(images)
        print(predicted)

    for epoch in range(num_epochs):
        # look at the bounding boxes in dataloader
        # for images, targets in data_loader:
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     for target in targets:
        #         bbox = target['boxes']
        #         idx = target['image_id']
        #         if bbox[0][0] > bbox[0][2] or bbox[0][1] > bbox[0][3]:
        #             print("Error: ", idx)
        #             print(bbox)
        #             raise NameError()
        #         else:
        #             print("OK: ", idx)

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # # evaluate on the train/test dataset
        print("Epoch: ", epoch)
        
        # for cat, iou in iou_results.items():
        #     print(cat, np.mean(iou))
    
    _, iou_results = evaluate(model, data_loader, device=device)
    print("Train: ", np.mean(iou_results[1]))
    _, iou_results = evaluate(model, data_loader_test, device=device)
    print("Test: ", np.mean(iou_results[1]))


if __name__ == '__main__':
    pipeline()