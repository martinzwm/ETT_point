import json
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchxrayvision as xrv
import torch 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def view_gt_bbox(
    root="/home/ec2-user/data/MIMIC_ETT_annotations", 
    annotation_file='annotations.json', 
    image_dir='PNGImages', target_dir='bbox3046'):
    # load annotations.json
    f = open(os.path.join(root, annotation_file))
    data = json.load(f)

    # image to id
    img_to_id = {}
    for img in data['images']:
        file_name = img['file_name'].replace('.dcm', '.png')
        img_to_id[file_name] = img['id']

    # id to ann
    id_to_ann = {}
    for i, ann in enumerate(data['annotations']):
        if ann['image_id'] not in id_to_ann:
            id_to_ann[ann['image_id']] = []
        id_to_ann[ann['image_id']].append(i)
    
    seg_model = xrv.baseline_models.chestx_det.PSPNet()

    # draw bbox
    for file_path in os.listdir( os.path.join(root, image_dir)):
        print(file_path)
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")

        ann_idx = id_to_ann[img_to_id[file_path]]
        bbox = None
        for idx in ann_idx:
            ann = data['annotations'][idx]
            if ann["category_id"] == 3046 or ann["category_id"] == 3047:
                xmin, ymin, w, h = ann["bbox"]
                xmax = xmin + w
                ymax = ymin + h
                bbox = [xmin, ymin, xmax, ymax]
                image = draw_single_bbox(image, bbox, seg_model)

        image.save(os.path.join(root, target_dir, file_path))


def draw_single_bbox(image, bbox, seg_model=None):
    # run segmentation model
    image = draw_trachea(image, seg_model)

    # draw bbox
    labelled_img = ImageDraw.Draw(image)
    shapes = [bbox]
    labelled_img.rectangle(shapes[0], outline="red", width = 1)
    return image


def draw_trachea(image, seg_model=None):
    image = np.array(image)
    orig_image = image[:, :, 0]
    image = xrv.datasets.normalize(image, 255) # convert 8-bit image to [-1024, 1024] range
    image = image.mean(2)[None, ...] # Make single color channel
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        pred = seg_model(image)
    # visualize as binary mask
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    mask = pred[0, 12].numpy()

    # Convert the grayscale image to RGB
    img_rgb = np.stack((orig_image, orig_image, orig_image), axis=-1)

    # Create a colored version of the mask with transparency
    color = np.array([0, 255, 0], dtype=np.uint8)  # Green color
    alpha = 0.2  # Transparency factor, between 0 (fully transparent) and 1 (fully opaque)
    mask_colored = np.stack((mask, mask, mask), axis=-1) * color * alpha

    # Overlay the mask on the original image
    result = np.where(mask_colored > 0, img_rgb * (1 - alpha) + mask_colored, img_rgb)

    # convert back to PIL image
    result = Image.fromarray(result.astype('uint8'), 'RGB')
    return result


def downsize(
    root="/home/ec2-user/data/MIMIC_ETT_annotations", 
    image_dir='PNGImages', target_dir='downsized'):
    resized_dim = 512
    # load target.json
    f = open(os.path.join(root, 'annotations_original.json'))
    data = json.load(f)

    # downsize images
    for file_path in os.listdir( os.path.join(root, image_dir)):
        print(file_path)
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # crop
        if w > h:
            image = image.crop(((w-h)/2, 0, (w+h)/2, h)) # cut evenly from left and right
        else:
            image = image.crop((0, (h-w)*0.2, w, h-(h-w)*0.8)) # cut more from bottom and less from top

        # resize
        image = image.resize((resized_dim, resized_dim))
        image.save(os.path.join(root, target_dir, file_path))

    # downsize gt bbox
    for i, ann in enumerate(data['annotations']):
        if ann['image_id'] == 2946144: # corrupted image
            continue
        if "bbox" not in ann:
            # print(ann)
            continue
        for i in range(len(data['images'])):
            if data['images'][i]['id'] == ann['image_id']:
                h = data['images'][i]['height']
                w = data['images'][i]['width']
                break
        curr_dim = min(w, h)
        if w > h:
            ann['bbox'][0] = ann['bbox'][0] - (w-h)/2
        else:
            ann['bbox'][1] = ann['bbox'][1] - (h-w)*0.2
        ann['bbox'][0] = ann['bbox'][0] * resized_dim / curr_dim
        ann['bbox'][1] = ann['bbox'][1] * resized_dim / curr_dim
        ann['bbox'][2] = ann['bbox'][2] * resized_dim / curr_dim
        ann['bbox'][3] = ann['bbox'][3] * resized_dim / curr_dim
    
    # save target.json
    with open(os.path.join(root, 'annotations_downsized_512.json'), 'w') as outfile:
        json.dump(data, outfile)


def normalize(
    root="/home/ec2-user/data/MIMIC-1105", 
    image_dir='downsized', target_dir='downsized_norm'
    ):
    clip_limit = 3.0
    tile_grid_size = (32, 32)

    for file_path in os.listdir(os.path.join(root, image_dir)):
        image_path = os.path.join(root, image_dir, file_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized_image = clahe.apply(gray_image)
        cv2.imwrite(os.path.join(root, target_dir, file_path), equalized_image)


def get_stats(root="/home/ec2-user/data/MIMIC_ETT_annotations", image_dir='PNGImages'):
    # find mean and std of images in folder
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for file_path in os.listdir( os.path.join(root, image_dir)):
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = image / 255
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))
    mean /= len(os.listdir( os.path.join(root, image_dir)))
    std /= len(os.listdir( os.path.join(root, image_dir)))
    print("Mean of the dataset: ", mean)
    print("Std of the dataset: ", std)


def generate_segmask(
    root="/home/ec2-user/data/MIMIC-1105", 
    image_dir='PNGImage', save_name='segmasks.json'
    ):
    segmask = {}
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to(DEVICE)

    # traverse through all images in directory
    ctr = 0
    for file_path in os.listdir(os.path.join(root, image_dir)):
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")

        image = np.array(image)
        image = xrv.datasets.normalize(image, 255) # convert 8-bit image to [-1024, 1024] range
        image = image.mean(2)[None, ...] # Make single color channel
        image = torch.from_numpy(image).unsqueeze_(0)
        image = image.to(DEVICE)
        
        pred = model(image).detach().cpu().numpy()
        pred = pred[0, 12, :, :]
        segmask[file_path] = pred.tolist()

        if ctr % 100 == 0:
            print(ctr)
        ctr += 1

    # save segmask to json
    with open(os.path.join(root, save_name), 'w') as outfile:
        json.dump(segmask, outfile)


if __name__ == "__main__":
    # downsize(root="/home/ec2-user/data/MIMIC-1105", image_dir='downsized', target_dir='downsized-512')
    # view_gt_bbox(
    #     root="/home/ec2-user/data/MIMIC-1105-512", 
    #     annotation_file='annotations-512.json', 
    #     image_dir='PNGImages', target_dir='bbox-512'
    #     )
    # normalize(root="/home/ec2-user/data/MIMIC-1105", image_dir='downsized', target_dir='downsized_norm')
    # get_stats(root="/home/ec2-user/data/MIMIC-1105-224", image_dir='downsized_norm')
    generate_segmask(root="/home/ec2-user/data/MIMIC-1105-512", image_dir='PNGImages', save_name='segmasks.json')