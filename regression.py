from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import argparse
from configparser import ConfigParser
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
import numpy as np
import os
import pandas as pd

from dataset import *
from pipeline_utils import *
from model_cxr import get_model


def get_dataloader():
    dataset_train = CXRDataset(
        root=arg.dataset_path, 
        image_dir='train',
        ann_file='anno_downsized.json',
        transforms=get_transform(train=True),
    )

    dataset_val = CXRDataset(
        root=arg.dataset_path, 
        image_dir='val',
        ann_file='anno_downsized.json',
        transforms=get_transform(train=False),
    )

    dataset_test = CXRDataset(
        root=arg.dataset_path, 
        image_dir='test',
        ann_file='anno_downsized.json',
        transforms=get_transform(train=False),
    )


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


def train(model, dataset_train, dataset_val, optimizer, device, epoch, model_1=None):
    model.train()
    if model_1:
        model_1.eval()
    best_loss = test(model, dataset_val, device, model_1)

    for i in range(epoch):
        print("Training epoch: ", i+1, " / ", epoch, " ...")
        
        for images, targets in dataset_train:
            predicted, centers, _ = forward(images, targets, model, device, model_1)
            if predicted is None:
                continue
            
            # Loss (3 components: carina, ETT, distance)
            loss = F.mse_loss(predicted[:, :4], centers)
            # predicted distance between carina and ETT
            dist = predicted[:, 4]
            # actual distance between carina and ETT
            dist_gt = torch.sqrt(torch.sum((centers[:, :2] - centers[:, 2:])**2, dim=1))
            loss += F.mse_loss(dist, dist_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Progress report
        print("Training set")
        train_loss = test(model, dataset_train, device, model_1)
        print("Validation set")
        val_loss = test(model, dataset_val, device, model_1)

        # Save best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), 
                "ckpts/{}_model{}_lr={}_bs={}.pt".format(
                    arg.backbone,
                    arg.model_num,
                    round(optimizer.param_groups[0]['lr'], 5),
                    arg.batch_size,
                    )
                )
        
        # Log to wandb
        if arg.logging:
            dst = log_images(model, dataset_val, device, model_1, r=5)
            wandb.log({
                "train/loss": train_loss, 
                "val/loss": val_loss,
                "images": wandb.Image(dst)
                })
    return val_loss
        

def test(model, dataset_test, device, model_1=None, save_to_csv=False):
    model.eval()
    if model_1 is not None:
        model_1.eval()
    carina_losses, ett_losses, dist_losses = [], [], []

    if save_to_csv:
        df = pd.DataFrame(columns=['image_id', 'category_id', 'x', 'y'])

    for images, targets in dataset_test:
        predicted, centers, kept_ids = forward(images, targets, model, device, model_1)
        if predicted is None:
                continue
        with torch.no_grad():
            # Carina and ETT loss
            carina_loss = F.mse_loss(predicted[:, :2], centers[:, :2])
            ett_loss = F.mse_loss(predicted[:, 2:4], centers[:, 2:4])
            carina_losses.append(torch.sqrt(carina_loss).item())
            ett_losses.append(torch.sqrt(ett_loss).item())

            # Distance loss
            # predicted distance between carina and ETT
            dist = predicted[:, 4]
            # actual distance between carina and ETT
            dist_gt = torch.sqrt(torch.sum((centers[:, :2] - centers[:, 2:4])**2, dim=1))
            loss = F.mse_loss(dist, dist_gt)
            dist_losses.append(torch.sqrt(loss).item())

            # Save to csv
            if save_to_csv:
                for i in range(len(predicted)): # note that we only consider images with ETT for now
                    # save carina coordinates
                    df = df.append({
                        'image_id': int(kept_ids[i]),
                        'category_id': 3046,
                        'x': predicted[i, 0].item(),
                        'y': predicted[i, 1].item()
                    }, ignore_index=True)

                    # save ETT coordinates
                    df = df.append({
                        'image_id': int(kept_ids[i]),
                        'category_id': 3047,
                        'x': predicted[i, 2].item(),
                        'y': predicted[i, 3].item()
                    }, ignore_index=True)


    carina_avg_loss = round( sum(carina_losses) / len(carina_losses), 1)
    print("\t carina L2 loss: ", carina_avg_loss)
    total_loss = carina_avg_loss
    
    ett_avg_loss = round( sum(ett_losses) / len(ett_losses), 1)
    print("\t ETT L2 loss: ", ett_avg_loss)
    total_loss += ett_avg_loss
    
    dist_avg_loss = round( sum(dist_losses) / len(dist_losses), 1)
    print("\t dist L2 loss: ", dist_avg_loss)
    total_loss += dist_avg_loss

    if save_to_csv:
        df.to_csv('predictions.csv', index=False)

    return total_loss


def classify_normal(model, dataset_test, device, model_1=None):
    model.eval()
    dists, dists_gt, dists_model = [], [], []
    for images, targets in dataset_test:
        predicted, centers, _ = forward(images, targets, model, device, model_1)
        with torch.no_grad():
            # predicted distance between carina and ETT
            dist = torch.sqrt(torch.sum((predicted[:, :2] - predicted[:, 2:4])**2, dim=1))
            dists.extend(dist.cpu().tolist())
            # distance directly output by model
            dists_model.extend(predicted[:, 4].cpu().tolist())
            # actual distance between carina and ETT
            dist_gt = torch.sqrt(torch.sum((centers[:, :2] - centers[:, 2:])**2, dim=1))
            dists_gt.extend(dist_gt.cpu().tolist())
    return dists, dists_gt, dists_model


def forward(images, targets, model, device, model_1=None):
    # Load image
    images = torch.stack([image for image in images], dim=0)
    images = images.to(device)

    # Load gt box
    bboxes = torch.stack([target['boxes'] for target in targets], dim=0)

    # Only keep the images that have ETT
    kept_indexes = [i for i in range(len(targets)) if targets[i]['labels'][1] != 0]
    kept_ids = [targets[i]['image_id_original'] for i in kept_indexes]
    if len(kept_ids) == 0:
        return None, None
    
    images = images[kept_indexes]
    bboxes = bboxes[kept_indexes]

    # Predict
    if model_1 is not None: # predict rough location of carina using model 1
        predicted = model_1(images)
        images, bboxes = crop_images(images, predicted, bboxes)
    predicted = model(images) # predict refined location of carina using model 2
    
    # gt center
    centers = torch.stack(
        [(bboxes[:, :, 0] + bboxes[:, :, 2]) / 2, (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2],
        dim=1
    )
    centers = centers.permute(0, 2, 1)
    centers = centers.reshape(centers.size(0), -1)
    centers = centers.to(device)

    return predicted, centers, kept_ids


def pipeline(evaluate=0):
    # Logistics
    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if arg.logging:
        wandb.init(project='ETT-debug')
    if arg.mode == "search":
        config = wandb.config
        arg.backbone = config['backbone']
        arg.lr = round(config['lr'], 5)
        arg.batch_size = config['batch_size']
        print(arg.backbone, arg.lr, arg.batch_size)

    # Load data
    dataloader_train, dataloader_val, dataloader_test = get_dataloader()
    
    # Load model
    if arg.backbone == "resnet":
        model = get_model(backbone=arg.backbone, model_num=arg.model_num, finetune=arg.finetune)
    else: # need to change the working directory to import from cxrlearn
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model = get_model(backbone=arg.backbone, model_num=arg.model_num, finetune=arg.finetune)
        os.chdir(os.path.join(os.getcwd(), ".."))
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    # Load model 1 if training model 2
    model_1 = None
    if arg.model_num == 2:
        if arg.model1_ckpt == None:
            raise ValueError("Need to provide the checkpoint for model 1 when training model 2")
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model_1 = get_model(backbone=arg.backbone, model_num=1)
        os.chdir(os.path.join(os.getcwd(), ".."))
        model_1.load_state_dict(torch.load(arg.model1_ckpt))
        model_1.to(device)

    # If we just want to evaluate the model, we don't need to train
    if evaluate == 1:
        print("Testing on test set ...")
        test_loss = test(model, dataloader_test, device, model_1, save_to_csv=True)
        return
    elif evaluate == 2:
        print("Testing on test set ...")
        dists, dists_gt, dists_model = classify_normal(model, dataloader_test, device, model_1)
        # save to csv
        df = pd.DataFrame({"dists": dists, "dists_gt": dists_gt, "dists_model": dists_model})
        df.to_csv("normal_vs_abnormal.csv")
        return

    # Train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    val_loss = train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, model_1)

    # Evaluate
    if arg.mode == "train":
        # Test on the best model so far
        print("Testing on test set ...")
        model.load_state_dict(torch.load(
            "ckpts/{}_model{}_lr={}_bs={}.pt".format(
                arg.backbone,
                arg.model_num,
                round(optimizer.param_groups[0]['lr'], 5),
                arg.batch_size,
                )
            ))
        test_loss = test(model, dataloader_test, device, model_1, save_to_csv=True)
        if arg.logging:
            wandb.log({"test/loss": test_loss})
            wandb.finish()
    elif arg.mode == "search":
        wandb.log({"val/loss": val_loss})
    else:
        raise ValueError("Invalid mode")


def hyperparameter_search():
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': 
        {
            # 'backbone': {'values': ['resnet', 'chexzero', "mococxr", "refers", "gloria"]},
            'backbone': {'values': ["gloria"]},
            'lr': {'distribution': 'log_uniform', 
                   'min': int(np.floor(np.log(1e-5))), 
                   'max': int(np.ceil(np.log(1e-3)))},
            'batch_size': {'values': [2, 4]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ETT-debug')
    wandb.agent(sweep_id, function=pipeline, count=20)
    wandb.finish()


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('--logging', type=int, default=1, help='Enable logging to wandb')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--finetune', type=int, default=0, help='Finetune the model')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--model_num', type=int, default=1, help='Model number')
    parser.add_argument('--model1_ckpt', type=str, default=None, help='Checkpoint path for model 1')
    parser.add_argument('--dataset_path', type=str, default='/home/ec2-user/data/MAIDA_RANZCR', help='Path for dataset')
    parser.add_argument('--mode', type=str, default='train', help='train: regular pipeline, search: hyperparameter search')
    parser.add_argument('--evaluate', type=int, default=0, help='Evaluation mode: 0: train and test, 1: test, 2: classify normal vs abnormal')
    parser.add_argument('--backbone', type=str, default='resnet', help='Pretrained backbone model')

    arg = parser.parse_args()
    arg.logging = True if arg.logging == 1 else False
    arg.finetune = True if arg.finetune == 1 else False

    if arg.mode == 'train':
        pipeline(arg.evaluate)
    elif arg.mode == 'search':
        hyperparameter_search()