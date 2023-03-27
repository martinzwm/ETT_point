import random
import torch
import math
from torchvision.transforms import functional as F
normalized = True
if normalized:
    MU, STD = 0.52561661, 0.20110909
else:
    MU, STD = 0.49714759, 0.21722301
    

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomTranslate(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, image, target):
        height, width = image.shape[-2:]
        pixels = int(min(height, width) * self.ratio)
        shift_x = random.randint(-pixels, pixels)
        shift_y = random.randint(-pixels, pixels)
        image = F.affine(image, translate=(shift_x, shift_y), angle=0, scale=1, shear=0, fill=-MU/STD)
        for i, bbox in enumerate(target["boxes"]):
            x, y, x_max, y_max = bbox
            new_bbox = [x + shift_x, y + shift_y, x_max + shift_x, y_max + shift_y]
            target["boxes"][i] = torch.tensor(new_bbox).view(1, 4)
        return image, target


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, target):
        angle = random.uniform(-self.angle, self.angle)
        image = F.rotate(image, angle, fill=-MU/STD)

        height, width = image.shape[-2:]
        center = (width / 2, height / 2)

        for i, bbox in enumerate(target["boxes"]):
            x, y, x_max, y_max = bbox
            corners = [(x, y), (x_max, y), (x_max, y_max), (x, y_max)]
            corners = [self.__rotate_point(c, center, angle) for c in corners]
            x, y = zip(*corners)
            new_bbox = [min(x), min(y), max(x), max(y)]
            target["boxes"][i] = torch.tensor(new_bbox).view(1, 4)
        return image, target

    def __rotate_point(self, point, center, angle):
        """Rotate a point around a center point by an angle."""
        x, y = point
        cx, cy = center
        radians = angle * math.pi / 180
        cos = math.cos(radians)
        sin = math.sin(radians)
        nx = (cos * (x - cx)) + (sin * (y - cy)) + cx
        ny = (cos * (y - cy)) - (sin * (x - cx)) + cy
        return nx, ny


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target
