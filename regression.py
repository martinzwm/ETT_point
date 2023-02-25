from __future__ import print_function

import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import wandb

from dataset import *


def get_dataloader():
    dataset_train = CXRDataset(
        root='/home/ec2-user/data/MIMIC_ETT_annotations', 
        image_dir='downsized',
        ann_file='annotations_downsized.json',
        transforms=get_transform(train=True)
        )
    dataset_val = deepcopy(dataset_train)
    dataset_val.transforms = get_transform(train=False)
    dataset_test = deepcopy(dataset_val)
    N = len(dataset_train)

    val_size, test_size = N // 3, N // 3
    indices = torch.randperm(N).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:val_size])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[val_size:val_size+test_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[val_size+test_size:])

    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=arg.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=arg.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=arg.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return dataloader_train, dataloader_val, dataloader_test


def get_model(finetune=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    if finetune==False:
        # Freeze backbone layers
        for param in model.parameters():
            param.requires_grad = False

    model = nn.Sequential(
        *[module for _, module in model.backbone.items()],
        nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 8, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(392, 2),
    )
    return model


def train(model, dataset_train, dataset_val, optimizer, device, epoch, logging=False, finetune=False):
    model.train()
    best_loss = test(model, dataset_val, device)

    for i in range(epoch):
        print("Training epoch: ", i+1, " / ", epoch, " ...")
        
        for images, targets in dataset_train:
            images = torch.stack([image for image in images], dim=0)
            images = images.to(device)
            predicted = model(images)

            bboxes = torch.stack([target['boxes'].squeeze(0) for target in targets], dim=0)
            centers = torch.stack(
                [(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2],
                dim=1
            )
            centers = centers.to(device)

            # Loss
            if arg.loss == "mse":
                loss = F.mse_loss(predicted, centers)
            elif arg.loss == "piecewise":
                # apply l2 loss if prediction is outside of bbox
                masks = [out_bbox(predicted[i], bboxes[i]) for i in range(predicted.size(0))]
                masks = torch.tensor(masks).to(device)
                loss = [torch.sum((predicted[i] - centers[i]) ** 2)  for i in range(predicted.size(0))]
                loss = torch.stack(loss, dim=0)
                loss = torch.sum(loss * masks) / (predicted.size(0) * predicted.size(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Progress report
        print("Training set", end=" ")
        train_loss = test(model, dataset_train, device)
        print("Validation set", end=" ")
        val_loss = test(model, dataset_val, device)

        # Save best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), 
                "ckpts/model_lr={}_{}.pt".format(
                    optimizer.param_groups[0]['lr'],
                    "finetune" if finetune else " ", 
                    )
                )
        
        # Log to wandb
        if logging:
            dst = log_images(model, dataset_val, device)
            wandb.log({
                "train/loss": train_loss, 
                "val/loss": val_loss,
                "images": wandb.Image(dst)
                })
        

def test(model, dataset_test, device):
    model.eval()
    dist = []

    for images, targets in dataset_test:
        images = torch.stack([image for image in images], dim=0)
        images = images.to(device)
        predicted = model(images)

        bboxes = torch.stack([target['boxes'].squeeze(0) for target in targets], dim=0)
        centers = torch.stack(
            [(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2],
            dim=1
        )
        centers = centers.to(device)

        with torch.no_grad():
            # L2 loss
            loss = F.mse_loss(predicted, centers)
            loss = torch.sqrt(loss)
            dist.append(loss.item())
    
    avg_loss = sum(dist) / len(dist)
    print("Average L2 loss: ", avg_loss)
    return avg_loss


def log_images(model, dataset, device):
    # Randomly select 3 images from the validation set, and plot the predicted center
    # and the ground truth center.
    indices = np.random.choice(len(dataset), size=3, replace=False)
    images = []
    
    for i in indices:
        image, target = dataset.dataset[i]
        image.unsqueeze_(0)
        image = image.to(device)
        predicted = model(image)
        predicted = predicted.cpu().detach().numpy()[0]

        gt_box = target['boxes'].squeeze(0).cpu().detach().numpy()
        gt = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]

        # Plot the image using PIL
        image = image.cpu().detach().numpy()[0]
        image = np.moveaxis(image, 0, -1)
        # unnormalize
        mean = np.array([0.49271007, 0.49271007, 0.49271007])
        std = np.array([0.23071574, 0.23071574, 0.23071574])
        image = std * image + mean
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.ellipse((predicted[0]-10, predicted[1]-10, predicted[0]+10, predicted[1]+10), fill='green')
        draw.ellipse((gt[0]-10, gt[1]-10, gt[0]+10, gt[1]+10), fill='red')
        print(gt, predicted)
        images.append(image)
    
    # Concatenate the images
    dst = Image.new(
        'RGB', 
        (images[0].width + images[1].width + images[2].width + 200, images[0].height+20)
        )
    dst.paste(images[0], (0, 10))
    dst.paste(images[1], (images[0].width + 100, 10))
    dst.paste(images[2], (images[0].width + + images[0].width + 200, 10))
    return dst
        

def out_bbox(predicted, bbox):
    # Check if the predicted center is inside the bounding box
    if predicted[0] >= bbox[0] and predicted[0] <= bbox[2] and \
        predicted[1] >= bbox[1] and predicted[1] <= bbox[3]:
        return 0
    else:
        return 1


def pipeline():
    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, dataloader_test = get_dataloader()
    finetune = True if arg.finetune == 1 else False
    model = get_model(finetune=finetune)
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    logging = True if arg.logging == 1 else False
    if logging:
        wandb.init(project='ETT_point')
        wandb.config = {}

    train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, logging, finetune)

    print("Testing on test set ...")
    test_loss = test(model, dataloader_test, device)

    if logging:
        wandb.log({"test/loss": test_loss})
        wandb.finish()


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('--logging', type=int, default=1, help='Enable logging to wandb')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--finetune', type=int, default=0, help='Finetune the model')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')

    arg = parser.parse_args()
    pipeline()