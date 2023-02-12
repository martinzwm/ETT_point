from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import transforms as T

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

    val_size, test_size = len(dataset) // 3, len(dataset) // 3
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, [len(dataset)-val_size-test_size, val_size, test_size]
        )

    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return dataloader_train, dataloader_val, dataloader_test


def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    
    model = nn.Sequential(
        *[module for _, module in model.backbone.items()],
        nn.Conv2d(2048, 1, kernel_size=(100, 100), stride=(1, 1)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(36 * 46, 2)
    )
    return model


def train(model, dataset_train, dataset_val, optimizer, device, epoch):
    model.train()
    best_loss = float('inf')

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

            # L2 loss
            loss = F.mse_loss(predicted, centers)
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
                "ckpts/model_epoch={}_lr={}.pt".format(i+1, optimizer.param_groups[0]['lr'])
                )
        


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


def pipeline(ckpt=None):
    torch.manual_seed(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_train, dataloader_val, dataloader_test = get_dataloader()
    model = get_model()
    if ckpt != None:
        model.load_state_dict(torch.load(ckpt))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    train(model, dataloader_train, dataloader_val, optimizer, device, 2)

    print("Testing on test set ...")
    test(model, dataloader_test, device)



if __name__ == "__main__":    
    pipeline()