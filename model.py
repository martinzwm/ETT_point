import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'cxrlearn'))

import torch
import torch.nn as nn
import cxrlearn.cxrlearn as cxrlearn


def get_model(backbone="resnet", model_num=1, finetune=False):
    if backbone == "resnet":
        model = get_resnet(model_num, finetune)
    elif backbone == "chexzero":
        model = get_chexzero(model_num, finetune)
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
        nn.Linear(8, 2)
    ]
    model = nn.Sequential(*modules)
    return model


def get_resnet(model_num=1, finetune=False):
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
            nn.Linear(128, 2)
            ])
    elif model_num == 2:
        modules.extend([
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
            ])
    
    model = nn.Sequential(*modules)
    return model