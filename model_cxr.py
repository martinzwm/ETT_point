import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'cxrlearn'))

import torch
import torch.nn as nn
import torchxrayvision as xrv
import cxrlearn.cxrlearn as cxrlearn
from model_utils import *
import matplotlib.pyplot as plt


def get_model(backbone="resnet", model_num=1, finetune=False):
    if backbone == "resnet":
        model = get_resnet(model_num=model_num, finetune=finetune)
    elif backbone == "chexzero":
        model = get_chexzero(model_num=model_num, finetune=finetune)
    elif backbone == "gloria":
        model = get_gloria(model_num=model_num, finetune=finetune)
    elif backbone == "CNN":
        model = get_CNN()
    elif backbone == "SAR":
        model = get_SAR()
    return model


def get_chexzero(model_num=1, finetune=False):
    """
    Backbone: ChexZero
    """
    model = cxrlearn.chexzero(
            freeze_backbone=(finetune==False), 
            linear_layer_dim=512, 
            num_out=14, # this is dummy b/c we're not using the last layer
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            )
    modules = [
        model[0],
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.BatchNorm1d(8),
        nn.Flatten(),
        nn.Linear(8, 5)
    ]
    model = nn.Sequential(*modules)
    return model


def get_SAR(model_num=1, finetune=False):
    """
    Backbone: SAR
    """
    model = SAR_point()
    return model


class SAR_point(nn.Module):
    def __init__(self):
        super(SAR_point, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Freeze segmentation backbone from torchxray vision
        self.backbone = xrv.baseline_models.chestx_det.PSPNet()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # For point detection
        self.point_1 = conv2DBatchNormRelu(
            in_channels=1, k_size=3, n_filters=32, padding=1, stride=2, bias=False
        )
        self.point_2 = conv2DBatchNormRelu(
            in_channels=32, k_size=3, n_filters=32, padding=1, stride=2, bias=False
        )
        self.point_3 = conv2DBatchNormRelu(
            in_channels=32, k_size=3, n_filters=16, padding=1, stride=2, bias=False
        )
        self.point_4 = conv2DBatchNormRelu(
            in_channels=16, k_size=3, n_filters=8, padding=1, stride=2, bias=False
        )
        self.point_fc = nn.Linear(8 * 4 * 4, 5)

    
    def preprocess(self, imgs):
        x = torch.zeros((len(imgs), 1, 512, 512))
        for i, img in enumerate(imgs):
            img = img.cpu().numpy() * 255
            img = xrv.datasets.normalize(img, 255)
            img = img.mean(2)[None, ...] # Make single color channel
            img = self.transform(img)
            img = torch.from_numpy(img)
            x[i] = img
        return x.to(self.device)
        

    def forward(self, img):
        img = self.preprocess(img)
        print(img.size())
        
        model = xrv.baseline_models.chestx_det.PSPNet()
        pred = model(img).detach().cpu().numpy()
        print(pred.shape)

        # plot using matplotlib and save to png
        plt.imshow(pred[0, 12, :, :])



        img = img.cpu().numpy()
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        print(img.shape)
        img = img.mean(1)[None, ...] # Make single color channel
        img = torch.from_numpy(img)
        print(img.size())
        raise NameError()
        
        model = xrv.baseline_models.chestx_det.PSPNet()
        pred = model(img).detach().cpu().numpy()
        print(pred.shape)

        # visualize as heatmap
        plt.imshow(pred.astype(np.uint8))
        plt.show()
        plt.savefig("mask.png")

        raise NameError()
    
        x = self.preprocess(x)
        x = self.backbone(x)
        # visualize mask and save to .png
        x = x.detach().cpu().numpy()
        x = 1 / (1 + np.exp(-x))  # sigmoid
        x[x < 0.5] = 0
        x[x > 0.5] = 1
        mask = x[0, 12]
        print(mask.shape)
        print(np.mean(mask), np.std(mask))
        plt.imshow(mask)
        plt.savefig("mask.png")
        raise NameError()

        x = x[:, 12, :, :].unsqueeze(1)
        x = self.point_1(x)
        x = self.point_2(x)
        x = self.point_3(x)
        x = self.point_4(x)
        x = x.view(x.size(0), -1)
        x = self.point_fc(x)
        return x


def get_gloria(model_num=1, finetune=False):
    """
    Backbone: GLORIA
    """
    model = cxrlearn.gloria(
        model="resnet50",
        freeze_backbone=(finetune==False),
        num_ftrs=2048, # not used
        num_out=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )
    backbone = list(model.children())[0]

    modules = [
        *list(backbone.children())[:-2],
    ]

    if model_num == 1:
        modules.extend([
        ResNetBlock(2048, 1024, 1, 2),
        ResNetBlock(1024, 512, 1, 2),
        ResNetBlock(512, 256, 1, 2),
        ResNetBlock(256, 128, 1, 2),
        nn.Flatten(),
        nn.Linear(128, 5)
        ])
    elif model_num == 2:
        modules.extend([
        nn.Conv2d(2048, 512, kernel_size=(2, 2), stride=(1, 1), padding="same"),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 128, kernel_size=(2, 2), stride=(1, 1), padding="same"),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 32, kernel_size=(2, 2), stride=(1, 1), padding="same"),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Flatten(),
        nn.Linear(32*2, 5)
        ])

    model = nn.Sequential(*modules)
    return model


def get_resnet(object="carina", model_num=1, finetune=False):
    """
    Backbone: ResNet50
    2 models to choose from:
        - model 1: get rough location of carina from the full image
        - model 2: get refined location of carina from cropped image based on model 1
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    if finetune==False:
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
    ]
    if model_num == 1:
        modules.extend([
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(128, 5)
            ])
    elif model_num == 2:
        modules.extend([
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 5)
            ])
    
    model = nn.Sequential(*modules)
    return model


def get_CNN():
    """
    Backbone: CNN recreated based on Kara et al. paper on cascaded CNN for ETT detection
    """
    model = nn.Sequential(
        ResNetBlock(1, 48, 1, 2),
        ResNetBlock(48, 56, 4, 2),
        ResNetBlock(56, 64, 4, 2),
        ResNetBlock(64, 80, 3, 2),
        ResNetBlock(80, 96, 3, 2),
        ResNetBlock(96, 112, 3, 2),
        ResNetBlock(112, 128, 3, 2),
        nn.Flatten(),
        nn.Linear(128*2*2, 5),
    )
    print(model)
    return model


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv1_layers, stride):
        super(ResNetBlock, self).__init__()
        
        self.num_conv1_layers = num_conv1_layers
        self.conv1_layers = nn.ModuleList()
        for i in range(num_conv1_layers):
            if i == 0:
                self.conv1_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                self.conv1_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv1_layers.append(nn.BatchNorm2d(out_channels))
            self.conv1_layers.append(nn.LeakyReLU(inplace=True))
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            

    def forward(self, x):
        residual = x

        for layer in self.conv1_layers:
            x = layer(x)
        out = self.conv2(x)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    

from dataset import * 
import numpy as np
import torchvision
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load image
    dataset = CXRDataset(
            root='/home/ec2-user/data/MIMIC-1105-512', 
            image_dir='PNGImages',
            ann_file='annotations-512.json',
            transforms=get_transform(train=False),
            )
    img_original = dataset[100][0]
    img_original = img_original * 255
    img_original = img_original.permute(1, 2, 0)
    img = np.array(img_original)

    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel

    img = torch.from_numpy(img).unsqueeze_(0)
    
    model = xrv.baseline_models.chestx_det.PSPNet()
    pred = model(img).detach().cpu().numpy()
    print(pred.shape)

    # plot using matplotlib and save to png
    plt.imshow(pred[0, 12, :, :])
    plt.savefig('test.png')