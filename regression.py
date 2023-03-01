from __future__ import print_function

import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
import numpy as np

from dataset import *
from pipeline_utils import *


def get_dataloader():
    dataset_train = CXRDataset(
        root=arg.dataset_path, 
        image_dir='downsized',
        ann_file='annotations_downsized.json',
        transforms=get_transform(train=True)
        )
    dataset_val = deepcopy(dataset_train)
    dataset_val.transforms = get_transform(train=False)
    dataset_test = deepcopy(dataset_val)
    N = len(dataset_train)

    val_size, test_size = int(N * 0.2), int(N * 0.2)
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


def get_model(model_num=1):
    """
    2 models to choose from:
        - model 1: get rough location of carina from the full image
        - model 2: get refined location of carina from cropped image based on model 1
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    if arg.finetune==False:
        # Freeze backbone layers
        for param in model.parameters():
            param.requires_grad = False

    modules = [
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
    ]
    if model_num == 1:
        modules.append(nn.Linear(392, 2))
    elif model_num == 2:
        modules.append(nn.Linear(72, 2))
    
    model = nn.Sequential(*modules)
    model.name = model_num
    return model


def train(model, dataset_train, dataset_val, optimizer, device, epoch, model_1=None):
    model.train()
    if model_1:
        model_1.eval()
    best_loss = test(model, dataset_val, device, model_1)

    for i in range(epoch):
        print("Training epoch: ", i+1, " / ", epoch, " ...")
        
        for images, targets in dataset_train:
            # Load images
            images = torch.stack([image for image in images], dim=0)
            images = images.to(device)

            # Load gt
            bboxes = torch.stack([target['boxes'].squeeze(0) for target in targets], dim=0)
            
            # Predict
            if model_1 is not None: # predict rough location of carina using model 1
                predicted = model_1(images)
                images, bboxes = crop_images(images, predicted, bboxes)
            predicted = model(images)

            # gt center
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
        train_loss = test(model, dataset_train, device, model_1)
        print("Validation set", end=" ")
        val_loss = test(model, dataset_val, device, model_1)

        # Save best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), 
                "ckpts/model{}_lr={}_bs={}_loss={}.pt".format(
                    arg.model_num,
                    optimizer.param_groups[0]['lr'],
                    arg.batch_size,
                    arg.loss,
                    )
                )
        
        # Log to wandb
        if arg.logging:
            dst = log_images(model, dataset_val, device, model_1)
            wandb.log({
                "train/loss": train_loss, 
                "val/loss": val_loss,
                "images": wandb.Image(dst)
                })
    return val_loss
        

def test(model, dataset_test, device, model_1=None):
    model.eval()
    if model_1 is not None:
        model_1.eval()
    dist = []

    for images, targets in dataset_test:
        # Load image
        images = torch.stack([image for image in images], dim=0)
        images = images.to(device)

        # Load gt box
        bboxes = torch.stack([target['boxes'].squeeze(0) for target in targets], dim=0)

        # Predict
        if model_1 is not None: # predict rough location of carina using model 1
            predicted = model_1(images)
            images, bboxes = crop_images(images, predicted, bboxes)
        predicted = model(images) # predict refined location of carina using model 2
        
        # gt center
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


def pipeline():
    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, dataloader_test = get_dataloader()
    
    model = get_model(model_num=arg.model_num)
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    if arg.logging:
        wandb.init(project='ETT-MIMIC-1105')
        wandb.config = {}

    model_1 = None
    if arg.model_num == 2:
        if arg.model1_ckpt == None:
            raise ValueError("Need to provide the checkpoint for model 1 when training model 2")
        model_1 = get_model(model_num=1)
        model_1.load_state_dict(torch.load(arg.model1_ckpt))
        model_1.to(device)

    train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, model_1)
    print("Testing on test set ...")
    test_loss = test(model, dataloader_test, device, model_1)

    if arg.logging:
        wandb.log({"test/loss": test_loss})
        wandb.finish()


def search_objective():
    wandb.init(project='ETT-MIMIC-1105')
    config = wandb.config
    print(config)
    arg.lr = config['lr']
    arg.batch_size = config['batch_size']
    arg.loss = config['loss']

    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, _ = get_dataloader()
    
    model = get_model(model_num=arg.model_num)
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    val_loss = train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, model_1=None)
    wandb.log({"loss": val_loss})


def hyperparameter_search():
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': 
        {
            'lr': {'distribution': 'log_uniform', 'min': np.log(1e-4), 'max': np.log(1e-2)},
            'batch_size': {'values': [2, 4, 8, 16]},
            'loss': {'values': ['mse', 'piecewise']},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ETT-MIMIC-1105')
    wandb.agent(sweep_id, function=search_objective, count=10)
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
    parser.add_argument('--model_num', type=int, default=1, help='Model number')
    parser.add_argument('--model1_ckpt', type=str, default=None, help='Checkpoint path for model 1')
    parser.add_argument('--dataset_path', type=str, default='/home/ec2-user/data/MIMIC_ETT_annotations', help='Path for dataset')
    parser.add_argument('--search', type=int, default=0, help='Hyperparameter search')

    arg = parser.parse_args()
    arg.logging = True if arg.logging == 1 else False
    arg.finetune = True if arg.finetune == 1 else False
    arg.search = True if arg.search == 1 else False
    
    if arg.search:
        hyperparameter_search()
    else:
        pipeline()