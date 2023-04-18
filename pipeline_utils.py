from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from copy import deepcopy
from dataset import view_img

from transforms import MU, STD


def log_images(model, dataset, device, model_1=None, r=10):
    # Randomly select 3 images from the validation set, and plot the predicted center
    # and the ground truth center.
    indices = np.random.choice(len(dataset), size=3, replace=False)
    images = []
    
    for i in indices:
        image, target = dataset.dataset[i]
        while target["labels"][1] != 1:
            i = np.random.choice(len(dataset), size=1, replace=False)[0]
            image, target = dataset.dataset[i]
        image = image.unsqueeze_(0).to(device)
        gt_box = target['boxes']
        if model_1 is not None:
            predicted = model_1(image)
            image, gt_box = crop_image(image.squeeze_(0), predicted.squeeze_(0), gt_box)
        predicted = model(image)
        predicted = predicted[:, :4]
        predicted = predicted.cpu().detach().numpy()
        predicted = predicted.reshape(2, -1)
        gt_box = gt_box.cpu().detach().numpy()
        gt = np.array([(gt_box[:, 0] + gt_box[:, 2]) / 2, (gt_box[:, 1] + gt_box[:, 3]) / 2])
        gt = gt.transpose()

        # Plot the image using PIL
        image = image.cpu().detach().numpy()[0]
        image = np.moveaxis(image, 0, -1)
        # unnormalize
        mean = np.array([MU, MU, MU])
        std = np.array([STD, STD, STD])
        image = std * image + mean
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        # Draw the predicted and ground truth center
        draw = ImageDraw.Draw(image)
        for i in range(predicted.shape[0]):
            if i == 0:
                draw.ellipse((predicted[i][0]-r, predicted[i][1]-r, predicted[i][0]+r, predicted[i][1]+r), fill='green')
            else:
                draw.ellipse((predicted[i][0]-r, predicted[i][1]-r, predicted[i][0]+r, predicted[i][1]+r), fill='blue')
        for i in range(gt.shape[0]):
            draw.ellipse((gt[i][0]-r, gt[i][1]-r, gt[i][0]+r, gt[i][1]+r), fill='red')
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
    H_new, W_new = int(H * 0.5), int(W * 0.25)
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

    if image.size(1) != H_new or image.size(2) != W_new:
        print("WARNING: Image size is not correct")
        image = image[:, :H_new, :W_new]

    # Transform the bbox according to the crop
    for box in bbox:
        box[0], box[2] = box[0] - x_min, box[2] - x_min
        box[1], box[3] = box[1] - y_min, box[3] - y_min
    
    return image, bbox