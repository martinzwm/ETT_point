from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from copy import deepcopy
from dataset import view_img

from transforms import MU, STD


def log_images(model, dataset, device, model_1=None):
    # Randomly select 3 images from the validation set, and plot the predicted center
    # and the ground truth center.
    indices = np.random.choice(len(dataset), size=3, replace=False)
    images = []
    
    for i in indices:
        image, target = dataset.dataset[i]
        image.unsqueeze_(0)
        image = image.to(device)
        gt_box = target['boxes']
        if model_1 is not None:
            predicted = model_1(image)
            image, gt_box = crop_images(image, predicted, target['boxes'])
        predicted = model(image)
        predicted = predicted.cpu().detach().numpy()[0]

        gt_box = gt_box.squeeze(0).cpu().detach().numpy()
        gt = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]

        # Plot the image using PIL
        image = image.cpu().detach().numpy()[0]
        image = np.moveaxis(image, 0, -1)
        # unnormalize
        mean = np.array([MU, MU, MU])
        std = np.array([STD, STD, STD])
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


def crop_images(images, points, bboxs):
    images_new, bboxs_new = [], []
    for i in range(len(images)):
        image, point, bbox = images[i], points[i], bboxs[i]
        image, bbox = crop_image(image, point, bbox)
        images_new.append(image)
        bboxs_new.append(bbox)
    images_new, bboxs_new = torch.stack(images_new, dim=0), torch.stack(bboxs_new, dim=0)
    return images_new, bboxs_new


def crop_image(image, point, bbox):
    # Crop the image around the point
    C, H, W = image.size()
    H_new, W_new = int(H * 0.5), int(W * 0.5)
    x_min, x_max = max(0, int(point[0] - W_new*0.5)), min(W, int(point[0] + W_new*0.5))
    y_min, y_max = max(0, int(point[1] - H_new*0.75)), min(H, int(point[1] + H_new*0.25))
    image = image[:, y_min:y_max, x_min:x_max]

    # Pad the image with color black if dimensions are not enough
    device = image.device
    if y_min == 0:
        pad = torch.zeros(C, H_new - image.size()[1], image.size()[2]).to(device) - MU/STD
        image = torch.cat((pad, image), dim=1)
    elif y_max == H:
        pad = torch.zeros(C, H_new - image.size()[1], image.size()[2]).to(device) - MU/STD
        image = torch.cat((image, pad), dim=1)
    if x_min == 0:
        pad = torch.zeros(C, image.size()[1], W_new - image.size()[2]).to(device) - MU/STD
        image = torch.cat((image, pad), dim=2)
    elif x_max == W:
        pad = torch.zeros(C, image.size()[1], W_new - image.size()[2]).to(device) - MU/STD
        image = torch.cat((pad, image), dim=2)

    # Transform the bbox according to the crop
    bbox = deepcopy(bbox)
    bbox[0], bbox[2] = bbox[0] - x_min, bbox[2] - x_min
    bbox[1], bbox[3] = bbox[1] - y_min, bbox[3] - y_min
    return image, bbox