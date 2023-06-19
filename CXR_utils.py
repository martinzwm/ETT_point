import json
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pydicom
import shutil
import pandas as pd

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
    error_ids = [
        4380029, 4380064, 4146693, 4379950, 4380433, 4380206, 4380559, 4146035, 4380713,
        4379767, 4379731, 4380006, 4380560, 4146328, 4380406, 4146445, 4146193, 2946144,
        ]
    for i, file_path in enumerate( os.listdir( os.path.join(root, image_dir)) ):
        print(i, file_path)
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")

        if img_to_id[file_path] in error_ids: # error
            # image.save("error_{}".format(file_path))
            continue

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
    # # run segmentation model
    # image = draw_trachea(image, seg_model)

    # draw bbox
    labelled_img = ImageDraw.Draw(image)
    shapes = [bbox]
    labelled_img.rectangle(shapes[0], outline="red", width = 2)
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


def dcm_to_png(
    root="/home/ec2-user/data/MAIDA",
    image_dir='data', target_dir='PNGImages'
    ):
    for file_path in os.listdir( os.path.join(root, image_dir)):
        print(file_path)
        image_path = os.path.join(root, image_dir, file_path)
        if image_path.endswith('.dcm'):
            ds = pydicom.dcmread(image_path)
            pixel_array_numpy = ds.pixel_array.astype(float)
            image = (np.maximum(pixel_array_numpy, 0) / pixel_array_numpy.max()) * 255.0
            image = np.uint8(image)
            if len(image.shape) == 2:  # If the image is grayscale, convert to RGB
                image = np.stack((image,) * 3, axis=-1) 
            image = Image.fromarray(image, 'RGB')
            file_path = file_path.replace('.dcm', '.png')
            image.save(os.path.join(root, target_dir, file_path))


def downsize(
    root="/home/ec2-user/data/MAIDA", 
    image_dir='PNGImages', target_dir='downsized',
    anno="anno.json", anno_downsized="anno_downsized.json"
    ):
    resized_dim = 512
    # load target.json
    f = open(os.path.join(root, anno))
    data = json.load(f)

    # downsize images
    for file_path in os.listdir(os.path.join(root, image_dir)):
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
    with open(os.path.join(root, anno_downsized), 'w') as outfile:
        json.dump(data, outfile)


def normalize(
    root="/home/ec2-user/data/MIMIC-1105", 
    image_dir='downsized', target_dir='downsized_norm'
    ):
    clip_limit = 3.0
    tile_grid_size = (8, 8)

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


def move_corrupted_images(
        root = "/home/ec2-user/data/MAIDA",
        image_dir = 'norm-512', target_dir = 'corrupted-512',
        error_ids = None,
        ):
        error_ids = [
            'a47262e8-f6091196-8e01bbee-05135ac6-64f49a96.png',
            'b935d184-c6a41da2-1b010e90-f2a0ee64-0071cd02.png',
            '0243ae7e-17b5ad32-6d1dcc4d-1b88104b-ef020f97.png',
            '40524a0f-86c821e2-e864cef6-df641195-736a9695.png',
            '34ad06d4-475863f1-f3712cec-783c3b99-308cf886.png',
            'ffa014fe-31d16e1a-3fb81bd7-80bf19d9-b7cba2c5.png',
            '674c5306-1692a472-a26817d0-5cee641f-06d8efb7.png',
            'a9042797-0293acb8-181d97fd-8db901a8-cad62048.png',
            'a5e92382-516aa78b-96e78b5c-add9edb9-fb669ad4.png',
            '09f8b265-36ff1b1a-59f9d43a-14fe4081-4fe45d36.png',
            '6bc2ffd4-cd939fe7-d8693f51-92938280-c0684822.png',
            '77307555-3c13553f-b0280559-2b144649-bdf9cd00.png',
            '0679acdd-91621f0c-b78b1cea-616a1b90-efd9a441.png',
            '78ced0ba-52b8ec30-e561e7f4-1cef117b-d04513b9.png',
            '014f4bbd-b58f8a76-7a2c26a7-0eb8c5ca-c4a99388.png',
            '09441900-ecc059ee-3b4ea545-8ef399ca-5ced7e0a.png',
            '2d92c458-4f3b6026-64864b22-0a4433d9-d68bd6a2.png',
            '3f33ebe6-6a1ddfb3-02d56727-44c3b635-abfa2730.png',
        ]

        for file_path in os.listdir(os.path.join(root, image_dir)):
            if file_path in error_ids:
                shutil.move(
                    os.path.join(root, image_dir, file_path),
                    os.path.join(root, target_dir, file_path)
                )


def combine_dataset(
        maida_folder="/home/ec2-user/data/MAIDA/norm-512",
        ranzcr_folder="/home/ec2-user/data/RANZCR/norm-512",
        csv_file="/home/ec2-user/ETT_Evaluation/data_split/all_data_split.csv",
        target_folder="/home/ec2-user/data/MAIDA_RANZCR",
    ):
    df = pd.read_csv(csv_file)
    error_ids = [
            'a47262e8-f6091196-8e01bbee-05135ac6-64f49a96',
            'b935d184-c6a41da2-1b010e90-f2a0ee64-0071cd02',
            '0243ae7e-17b5ad32-6d1dcc4d-1b88104b-ef020f97',
            '40524a0f-86c821e2-e864cef6-df641195-736a9695',
            '34ad06d4-475863f1-f3712cec-783c3b99-308cf886',
            'ffa014fe-31d16e1a-3fb81bd7-80bf19d9-b7cba2c5',
            '674c5306-1692a472-a26817d0-5cee641f-06d8efb7',
            'a9042797-0293acb8-181d97fd-8db901a8-cad62048',
            'a5e92382-516aa78b-96e78b5c-add9edb9-fb669ad4',
            '09f8b265-36ff1b1a-59f9d43a-14fe4081-4fe45d36',
            '6bc2ffd4-cd939fe7-d8693f51-92938280-c0684822',
            '77307555-3c13553f-b0280559-2b144649-bdf9cd00',
            '0679acdd-91621f0c-b78b1cea-616a1b90-efd9a441',
            '78ced0ba-52b8ec30-e561e7f4-1cef117b-d04513b9',
            '014f4bbd-b58f8a76-7a2c26a7-0eb8c5ca-c4a99388',
            '09441900-ecc059ee-3b4ea545-8ef399ca-5ced7e0a',
            '2d92c458-4f3b6026-64864b22-0a4433d9-d68bd6a2',
            '3f33ebe6-6a1ddfb3-02d56727-44c3b635-abfa2730',
            '1f808e34-141acb56-8d53cbff-36d59896-ed88b35f',
            "1.2.826.0.1.3680043.8.498.16515999586137292214341617674809427578",
        ]
    
    for index, row in df.iterrows():
        source = row['Source']
        file_name = row['FileName']
        split = row['Split']

        # Skip the error ids
        if file_name in error_ids:
            continue
        
        # Determine the source folder
        if source == 'MIMIC':
            source_folder = maida_folder
            if not file_name.endswith('.png'):
                file_name += '.png'
        elif source == 'RANZCR':
            source_folder = ranzcr_folder
            if not file_name.endswith('.jpg'):
                file_name += '.jpg'
        else:
            print(f"Unknown source {source} for file {file_name}")
            continue
        
        # Determine the destination folder based on the 'Split' column
        dest_folder = os.path.join(target_folder, split)
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copyfile(src_path, dest_path)

        if index % 100 == 0:
            print("Completed {}".format(index))


def combine_jsons(
        anno_maida="/home/ec2-user/data/MAIDA/anno_downsized.json",
        anno_ranzcr="/home/ec2-user/data/RANZCR/anno_downsized.json",
        anno_combined="/home/ec2-user/data/MAIDA_RANZCR/anno_downsized.json",
    ):
    with open(anno_maida, 'r') as f:
        maida = json.load(f)

    with open(anno_ranzcr, 'r') as f:
        ranzcr = json.load(f)

    combined = maida
    combined['images'] += ranzcr['images']
    combined['annotations'] += ranzcr['annotations']

    with open(anno_combined, 'w') as f:
        json.dump(combined, f)
    

if __name__ == "__main__":
    # dcm_to_png(
    #     root="/home/ec2-user/data/MAIDA",
    #     image_dir='data', target_dir='PNGImages'
    #     )

    # downsize(
    #     root="/home/ec2-user/data/RANZCR", 
    #     image_dir='PNGImages', target_dir='downsized',
    #     anno='RANZCR-ETT-annotations.json', anno_downsized='anno_downsized.json',
    #     )

    # view_gt_bbox(
    #     root="/home/ec2-user/data/RANZCR", 
    #     annotation_file='anno_downsized.json', 
    #     image_dir='downsized-512', target_dir='bbox-512'
    #     )

    # move_corrupted_images(
    #     root = "/home/ec2-user/data/MAIDA",
    #     image_dir = 'norm-512', target_dir = 'corrupted-512',
    #     error_ids = None,
    # )

    # normalize(root="/home/ec2-user/data/RANZCR", image_dir='downsized-512', target_dir='norm-512')
    
    get_stats(root="/home/ec2-user/data/MAIDA_RANZCR", image_dir='train')
    
    # generate_segmask(root="/home/ec2-user/data/MIMIC-1105-512", image_dir='PNGImages', save_name='segmasks.json')

    # combine_dataset(
    #     maida_folder="/home/ec2-user/data/MAIDA/norm-512", 
    #     ranzcr_folder="/home/ec2-user/data/RANZCR/norm-512", 
    #     csv_file="/home/ec2-user/ETT_Evaluation/data_split/all_data_split.csv", 
    #     target_folder="/home/ec2-user/data/MAIDA_RANZCR"
    #     )

    # combine_jsons(
    #     anno_maida="/home/ec2-user/data/MAIDA/anno_downsized.json",
    #     anno_ranzcr="/home/ec2-user/data/RANZCR/anno_downsized.json",
    #     anno_combined="/home/ec2-user/data/MAIDA_RANZCR/anno_downsized.json",
    # )