from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import transforms as T

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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


def get_dataloader():
    dataset = CXRDataset(
        root='/home/ec2-user/data/MIMIC_ETT_annotations', 
        image_dir='downsized',
        ann_file='annotations_downsized.json',
        target_file='target_downsized.json',
        transforms=get_transform(train=True)
        )
    dataset_test = CXRDataset(
        root='/home/ec2-user/data/MIMIC_ETT_annotations', 
        image_dir='downsized',
        ann_file='annotations_downsized.json',
        target_file='target_downsized.json',
        transforms=get_transform(train=True)
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

    return data_loader, data_loader_test


def get_model():
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    
    model = nn.Sequential(
        *[module for _, module in model.backbone.items()],
        nn.Conv2d(2048, 1, kernel_size=(100, 100), stride=(1, 1)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(36 * 46, 2)
    )
    return model


def train(model, dataset, optimizer, device, epoch):
    model.train()
    for i in range(epoch):
        print("Training epoch: ", i+1, " / ", epoch, " ...")
        
        for images, targets in dataset:
            images = images[0].unsqueeze(0)
            images = images.to(device)
            predicted = model(images)

            bbox = targets[0]['boxes'].squeeze(0)
            center = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]).unsqueeze(0)
            center = center.to(device)

            # L2 loss
            loss = F.mse_loss(predicted, center)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, dataset, device):
    model.eval()
    dist = []
    for images, targets in dataset:
        images = images[0].unsqueeze(0)
        images = images.to(device)
        predicted = model(images)

        bbox = targets[0]['boxes'].squeeze(0)
        center = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]).unsqueeze(0)
        center = center.to(device)

        loss = F.mse_loss(predicted, center)
        loss = torch.sqrt(loss)
        dist.append(loss.item())
    
    print("Average L2 loss: ", sum(dist) / len(dist))


def pipeline():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader, data_loader_test = get_dataloader()
    model = get_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    train(model, data_loader, optimizer, device, 10)

    print("Testing on training set ...")
    test(model, data_loader, device)
    print("Testing on test set ...")
    test(model, data_loader_test, device)


if __name__ == "__main__":    
    pipeline()