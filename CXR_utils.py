import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

    # draw bbox
    for file_path in os.listdir( os.path.join(root, image_dir)):
        print(file_path)
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")

        ann_idx = id_to_ann[img_to_id[file_path]]
        bbox = None
        for idx in ann_idx:
            ann = data['annotations'][idx]
            if ann["category_id"] == 3046:
                xmin, ymin, w, h = ann["bbox"]
                xmax = xmin + w
                ymax = ymin + h
                bbox = [xmin, ymin, xmax, ymax]

        image = draw_single_bbox(image, bbox)
        image.save(os.path.join(root, target_dir, file_path))


def draw_single_bbox(image, bbox):
    labelled_img = ImageDraw.Draw(image)
    shapes = [bbox]
    labelled_img.rectangle(shapes[0], outline="red", width = 10)
    return image


def downsize(
    root="/home/ec2-user/data/MIMIC_ETT_annotations", 
    image_dir='PNGImages', target_dir='downsized'):
    resized_dim = 1024
    # load target.json
    f = open(os.path.join(root, 'annotations.json'))
    data = json.load(f)

    # downsize images
    for file_path in os.listdir( os.path.join(root, image_dir)):
        print(file_path)
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        h, w = image.size

        # crop
        if h > w:
            image = image.crop(((h-w)/2, 0, (h+w)/2, w)) # cut evenly from left and right
        else:
            image = image.crop((0, (w-h)*0.2, h, w-(w-h)*0.8)) # cut more from bottom and less from top

        # resize
        image = image.resize((resized_dim, resized_dim))
        image.save(os.path.join(root, "downsized", file_path))

    # downsize gt bbox
    for i, ann in enumerate(data['annotations']):
        if ann['image_id'] == 2946144: # corrupted image
            continue
        if "bbox" not in ann:
            print(ann)
            continue
        for i in range(len(data['images'])):
            if data['images'][i]['id'] == ann['image_id']:
                w = data['images'][i]['height']
                h = data['images'][i]['width']
                break
        ann['bbox'][0] = ann['bbox'][0] * resized_dim / h
        ann['bbox'][1] = ann['bbox'][1] * resized_dim / w
        ann['bbox'][2] = ann['bbox'][2] * resized_dim / h
        ann['bbox'][3] = ann['bbox'][3] * resized_dim / w
    
    # save target.json
    with open(os.path.join(root, 'annotations_downsized.json'), 'w') as outfile:
        json.dump(data, outfile)


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
    

if __name__ == "__main__":
    # downsize(root="/home/ec2-user/data/MIMIC-981", image_dir='PNGImages', target_dir='downsized')
    view_gt_bbox(
        root="/home/ec2-user/data/MIMIC-981", 
        annotation_file='annotations_downsized.json', 
        image_dir='downsized', target_dir='bbox3046_downsized')
    # get_stats(root="/home/ec2-user/data/MIMIC-981", image_dir='downsized')
    