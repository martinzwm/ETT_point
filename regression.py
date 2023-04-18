from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import argparse
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
from model import get_model


def get_dataloader():
    dataset_train = CXRDataset(
        root=arg.dataset_path, 
        image_dir='downsized_norm',
        ann_file='annotations_downsized.json',
        transforms=get_transform(train=True)
        )
    dataset_val = deepcopy(dataset_train)
    dataset_val.transforms = get_transform(train=False)
    dataset_test = deepcopy(dataset_val)
    N = len(dataset_train)

    val_size, test_size = int(N * 0.2), int(N * 0.2)
    indices = torch.randperm(N).tolist()
    # dataset_train = torch.utils.data.Subset(dataset_train, indices[:val_size])
    # dataset_val = torch.utils.data.Subset(dataset_val, indices[val_size:val_size+test_size])
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:val_size+test_size])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[(N-test_size):])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[(N-test_size):])

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
            predicted, centers = forward(images, targets, model, device, model_1)
            if predicted is None:
                continue
            
            # Loss
            loss = F.mse_loss(predicted[:, :4], centers)
            if arg.loss == "distance":
                # predicted distance between carina and ETT
                # dist = torch.sqrt(torch.sum((predicted[:, :2] - predicted[:, 2:])**2, dim=1))
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
                "ckpts/{}_{}_model{}_lr={}_bs={}_loss={}.pt".format(
                    arg.object,
                    arg.backbone,
                    arg.model_num,
                    round(optimizer.param_groups[0]['lr'], 5),
                    arg.batch_size,
                    arg.loss,
                    )
                )
        
        # Log to wandb
        if arg.logging:
            dst = log_images(model, dataset_val, device, model_1, object=arg.object, r=2)
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
    carina_losses, ett_losses, dist_losses = [], [], []

    for images, targets in dataset_test:
        predicted, centers = forward(images, targets, model, device, model_1)
        if predicted is None:
                continue
        with torch.no_grad():
            # L2 loss
            if arg.object == "carina":
                carina_loss = F.mse_loss(predicted, centers)
                carina_losses.append(torch.sqrt(carina_loss).item())
            elif arg.object == "both":
                carina_loss = F.mse_loss(predicted[:, :2], centers[:, :2])
                ett_loss = F.mse_loss(predicted[:, 2:4], centers[:, 2:4])
                carina_losses.append(torch.sqrt(carina_loss).item())
                ett_losses.append(torch.sqrt(ett_loss).item())

            if arg.loss == "distance":
                # predicted distance between carina and ETT
                # dist = torch.sqrt(torch.sum((predicted[:, :2] - predicted[:, 2:])**2, dim=1))
                dist = predicted[:, 4]
                # actual distance between carina and ETT
                dist_gt = torch.sqrt(torch.sum((centers[:, :2] - centers[:, 2:4])**2, dim=1))
                loss = F.mse_loss(dist, dist_gt)
                dist_losses.append(torch.sqrt(loss).item())
            
    carina_avg_loss = round( sum(carina_losses) / len(carina_losses), 1)
    print("\t carina L2 loss: ", carina_avg_loss)
    total_loss = carina_avg_loss
    if arg.object == "both":
        ett_avg_loss = round( sum(ett_losses) / len(ett_losses), 1)
        print("\t ETT L2 loss: ", ett_avg_loss)
        total_loss += ett_avg_loss
        # print("\t Total detection loss: ", total_loss)
    
    if arg.loss == "distance":
        dist_avg_loss = round( sum(dist_losses) / len(dist_losses), 1)
        print("\t dist L2 loss: ", dist_avg_loss)
        total_loss += dist_avg_loss

    return total_loss


def classify_normal(model, dataset_test, device, model_1=None):
    model.eval()
    dists, dists_gt, dists_model = [], [], []
    for images, targets in dataset_test:
        predicted, centers = forward(images, targets, model, device, model_1)
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


def inference(test_img_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
    model = get_model(backbone=arg.backbone, object=arg.object, model_num=arg.model_num, finetune=arg.finetune)
    os.chdir(os.path.join(os.getcwd(), ".."))
    model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)
    model.eval()

    # load transform
    transform = get_transform(train=False)

    # load test images and predict
    result = []
    ctr = 0
    for img_name in os.listdir(test_img_folder):
        ctr += 1
        if ctr % 100 == 0:
            print(ctr)
        # prediction
        img_path = os.path.join(test_img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img, _ = transform(img, None)
        img = img.unsqueeze(0).to(device)
        predicted = model(img)
        predicted = predicted.cpu().detach().numpy()

        # save result
        carina = predicted[0, :2]
        ett = predicted[0, 2:4]
        dist = predicted[0, 4]
        dist1 = np.sqrt(np.sum((carina - ett)**2))
        result.append([img_name, carina[0], carina[1], ett[0], ett[0], dist, dist1])

        # Draw result
        img = img.cpu().detach().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([MU, MU, MU])
        std = np.array([STD, STD, STD])
        img = std * img + mean
        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        r = 3
        draw.ellipse((carina[0]-r, carina[1]-r, carina[0]+r, carina[1]+r), fill='green')
        draw.ellipse((ett[0]-r, ett[1]-r, ett[0]+r, ett[1]+r), fill='blue')
        img_save_path = os.path.join("/home/ec2-user/data/ranzcr/inference", img_name)
        img.save(img_save_path)
    
    # save result
    df = pd.DataFrame(result, columns=["image_name", "carina_x", "carina_y", "ett_x", "ett_y", "dist", "dist1"])
    df.to_csv("inference.csv", index=False)


def forward(images, targets, model, device, model_1=None):
    # Load image
    images = torch.stack([image for image in images], dim=0)
    images = images.to(device)

    # Load gt box
    bboxes = torch.stack([target['boxes'] for target in targets], dim=0)

    # Only keep the images that have ETT
    to_remove = [i for i in range(len(targets)) if targets[i]['labels'][1] == 0]
    if len(to_remove) == len(targets):
        return None, None
    
    for i in to_remove:
        images = torch.cat((images[:i], images[i+1:]), dim=0)
        bboxes = torch.cat((bboxes[:i], bboxes[i+1:]), dim=0)

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
    
    if arg.object == "carina":
        centers = centers[:, 0, :]
    elif arg.object == "both":
        centers = centers.reshape(centers.size(0), -1)
    else:
        raise ValueError("Invalid object type, needs to be either carina or both")
    centers = centers.to(device)

    return predicted, centers


def pipeline(evaluate=False):
    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, dataloader_test = get_dataloader()
    
    if arg.backbone == "resnet":
        model = get_model(backbone=arg.backbone, object=arg.object, model_num=arg.model_num, finetune=arg.finetune)
    else: # need to change the working directory to import from cxrlearn
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model = get_model(backbone=arg.backbone, object=arg.object, model_num=arg.model_num, finetune=arg.finetune)
        os.chdir(os.path.join(os.getcwd(), ".."))
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    model_1 = None
    if arg.model_num == 2:
        if arg.model1_ckpt == None:
            raise ValueError("Need to provide the checkpoint for model 1 when training model 2")
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model_1 = get_model(backbone=arg.backbone, object=arg.object, model_num=1)
        os.chdir(os.path.join(os.getcwd(), ".."))
        model_1.load_state_dict(torch.load(arg.model1_ckpt))
        model_1.to(device)

    if evaluate == 1:
        print("Testing on test set ...")
        test_loss = test(model, dataloader_test, device, model_1)
        return
    elif evaluate == 2:
        print("Testing on test set ...")
        dists, dists_gt, dists_model = classify_normal(model, dataloader_test, device, model_1)
        # save to csv
        df = pd.DataFrame({"dists": dists, "dists_gt": dists_gt, "dists_model": dists_model})
        df.to_csv("normal_vs_abnormal.csv")
        return
        

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    if arg.logging:
        # wandb.init(project='ETT-MIMIC-1105-224')
        wandb.init(project='ETT-debug')
        wandb.config = {}

    train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, model_1)
    print("Testing on test set ...")
    model.load_state_dict(torch.load(
        "ckpts/{}_{}_model{}_lr={}_bs={}_loss={}.pt".format(
            arg.object,
            arg.backbone,
            arg.model_num,
            round(optimizer.param_groups[0]['lr'], 5),
            arg.batch_size,
            arg.loss,
            )
        ))
    test_loss = test(model, dataloader_test, device, model_1)

    if arg.logging:
        wandb.log({"test/loss": test_loss})
        wandb.finish()


def search_objective():
    wandb.init(project='ETT-debug')
    config = wandb.config
    arg.backbone = config['backbone']
    arg.lr = round(config['lr'], 5)
    arg.batch_size = config['batch_size']
    arg.loss = config['loss']
    print(arg.backbone, arg.lr, arg.batch_size, arg.loss)

    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, _ = get_dataloader()

    if arg.backbone == "resnet":
        model = get_model(backbone=arg.backbone, object=arg.object, model_num=arg.model_num, finetune=arg.finetune)
    else: # need to change the working directory to import from cxrlearn
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model = get_model(backbone=arg.backbone, object=arg.object, model_num=arg.model_num, finetune=arg.finetune)
        os.chdir(os.path.join(os.getcwd(), ".."))
    if arg.ckpt != None:
        model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)

    model_1 = None
    if arg.model_num == 2:
        if arg.model1_ckpt == None:
            raise ValueError("Need to provide the checkpoint for model 1 when training model 2")
        os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
        model_1 = get_model(backbone=arg.backbone, object=arg.object, model_num=1)
        os.chdir(os.path.join(os.getcwd(), ".."))
        model_1.load_state_dict(torch.load(arg.model1_ckpt))
        model_1.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=arg.lr)
    val_loss = train(model, dataloader_train, dataloader_val, optimizer, device, arg.epoch, model_1=model_1)
    wandb.log({"loss": val_loss})


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
            'batch_size': {'values': [2, 4, 8, 16]},
            'loss': {'values': ['distance']},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ETT-debug')
    wandb.agent(sweep_id, function=search_objective, count=10)
    wandb.finish()


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('--logging', type=int, default=1, help='Enable logging to wandb')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--finetune', type=int, default=0, help='Finetune the model')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--model_num', type=int, default=1, help='Model number')
    parser.add_argument('--model1_ckpt', type=str, default=None, help='Checkpoint path for model 1')
    parser.add_argument('--dataset_path', type=str, default='/home/ec2-user/data/MIMIC-1105-224', help='Path for dataset')
    parser.add_argument('--mode', type=int, default=0, help='0: regular pipeline, 1: hyperparameter search, 2: inference')
    parser.add_argument('--evaluate', type=int, default=0, help='Evaluation mode: 0: train and test, 1: test, 2: classify normal vs abnormal')
    parser.add_argument('--backbone', type=str, default='resnet', help='Pretrained backbone model')
    parser.add_argument('--object', type=str, default='carina', help='Detect carina or both carina and ETT')

    arg = parser.parse_args()
    arg.logging = True if arg.logging == 1 else False
    arg.finetune = True if arg.finetune == 1 else False

    if arg.mode == 0:
        pipeline(arg.evaluate)
    elif arg.mode == 1:
        hyperparameter_search()
    elif arg.mode == 2:
        inference(arg.dataset_path)