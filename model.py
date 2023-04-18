import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'cxrlearn'))

import torch
import torch.nn as nn
import torchvision
import cxrlearn.cxrlearn as cxrlearn


def get_model(backbone="resnet", model_num=1, finetune=False):
    if backbone == "resnet":
        model = get_resnet(model_num=model_num, finetune=finetune)
    elif backbone == "chexzero":
        model = get_chexzero(model_num=model_num, finetune=finetune)
    elif backbone == "mococxr":
        model = get_mococxr( model_num=model_num, finetune=finetune)
    elif backbone == "refers":
        model = get_refers(model_num=model_num, finetune=finetune)
    elif backbone == "gloria":
        model = get_gloria(model_num=model_num, finetune=finetune)
    elif backbone == "CNN":
        model = get_CNN()
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
        nn.Linear(8, 4)
    ]
    model = nn.Sequential(*modules)
    return model


def get_mococxr(model_num=1, finetune=False):
    """
    Backbone: MocoCXR
    """
    model = cxrlearn.mococxr(
        model="resnet50",
        freeze_backbone=(finetune==False),
        num_out=14, # this is dummy b/c we're not using the last layer
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    modules = [
        *list(model.children())[:-2],
        nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Flatten(),
        nn.Linear(32*7*7, 4)
    ]
    model = nn.Sequential(*modules)
    return model


def get_refers(model_num=1, finetune=False):
    """
    Backbone: REFERS
    """
    model = cxrlearn.refers(
        freeze_backbone=(finetune==False),
        num_out=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    modules = [
        model,
        nn.Linear(768, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.BatchNorm1d(8),
        nn.Flatten(),
        nn.Linear(8, 4)
    ]
    model = nn.Sequential(*modules)
    return model


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
        nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(128),
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
        nn.Linear(32*4*2, 5)
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

