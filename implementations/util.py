import os
from collections import Counter
import numpy as np

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.autograd import Variable
import torch

import os
import os.path
from typing import Dict, List, Tuple

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def is_valid_file(filepath: str):
    # Add your custom logic here to determine if the file is valid
    # For example, you might want to check file extensions or other criteria
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".png")
    filename = os.path.basename(filepath)

    return (not filename.startswith(".")) and filename.lower().endswith(valid_extensions)


def custom_preprocessing(opt):
    transform = transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels),
        ]
    )
    return transform


def is_valid_class_generator(class_names: list[str]):

    def is_valid_class(filepath: str):
        parent_dir = os.path.basename(os.path.dirname(filepath))

        return is_valid_file(filepath) and (parent_dir in class_names)

    return is_valid_class


def get_number_instances_to_aug(dataset: ImageFolder):
    class_counts = Counter(dataset.targets)

    max_count = max(class_counts.values())

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    to_aug = {
        cls: {
            "images": max_count - count,
            "class": idx_to_class[cls],
            "is_valid_func": is_valid_class_generator([idx_to_class[cls]]),
        }
        for cls, count in class_counts.items()
        if count < max_count
    }

    to_aug["all"] = {
        "is_valid_func": is_valid_class_generator([idx_to_class[cls] for cls in to_aug.keys()]),
    }

    return to_aug


def save_images(images, path):
    if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for idx, img in enumerate(images):
        save_image(img.data, os.path.join(path, f"{idx:06}.png"), normalize=True)

def sample_image(
    to_aug,
    latent_dim,
    generator,
    results_path,
    all_labels=True,
    one_label: str=None,
):
    for label, data in to_aug.items():
        if label == "all":
            continue

        if not all_labels and one_label != data["class"]:
            continue

        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (data["images"], latent_dim))))

        if all_labels:
            labels = np.array([label for _ in range(data["images"])])
            labels = Variable(LongTensor(labels))

            gen_imgs = generator(z, labels)
        else:
            gen_imgs = generator(z)

        # get the path of the current python file, and save images relative to that path
        images_dir = os.path.join(results_path, data["class"])

        save_images(gen_imgs, images_dir)


class ImageFolderFilterClasses(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        filter_classes=None,
    ):
        self.filter_classes = filter_classes

        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        classes = [
            cls_name
            for cls_name in classes
            if (not self.filter_classes) or (cls_name in self.filter_classes)
        ]

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
