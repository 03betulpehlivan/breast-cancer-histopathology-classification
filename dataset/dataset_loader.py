# =====================================
# IMPROVED DATASET LOADER
# =====================================

import os
from PIL import Image

import torch

from torch.utils.data import (

    Dataset,

    DataLoader
)

from torchvision import transforms

from configs.config import CFG


import random

# =====================================
# TRAIN TRANSFORMS
# =====================================

train_transform = transforms.Compose([

    # =========================
    # RESIZE
    # =========================

    transforms.RandomResizedCrop(

       CFG.IMAGE_SIZE,

       scale=(0.90, 1.0)
    ),

    # =========================
    # AUGMENTATION
    # =========================

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=25
    ),

     transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.95, 1.05),
     ),

    transforms.ColorJitter(

        brightness=0.10,

        contrast=0.10,

        saturation=0.10,

        hue=0.02
    ),

    transforms.GaussianBlur(
         kernel_size=3
    ),


    # =========================
    # TENSOR
    # =========================

    transforms.ToTensor(),

    # =========================
    # NORMALIZATION
    # =========================

    transforms.Normalize(

        mean=CFG.IMAGE_MEAN,

        std=CFG.IMAGE_STD
    )
])


# =====================================
# VALIDATION / TEST TRANSFORMS
# =====================================

val_transform = transforms.Compose([

    transforms.Resize(
        (
            CFG.IMAGE_SIZE,
            CFG.IMAGE_SIZE
        )
    ),

    transforms.CenterCrop(
             CFG.IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=CFG.IMAGE_MEAN,

        std=CFG.IMAGE_STD
    )
])


# =====================================
# CUSTOM DATASET
# =====================================

class BreastCancerDataset(Dataset):

    def __init__(

        self,

        image_paths,

        labels,

        transform=None
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.transform = transform

    # =================================
    # LENGTH
    # =================================

    def __len__(self):

        return len(
            self.image_paths
        )

    # =================================
    # GET ITEM
    # =================================

    def __getitem__(

        self,

        idx
    ):

        image_path = self.image_paths[idx]

        label = self.labels[idx]

        # =========================
        # LOAD IMAGE
        # =========================

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"\nERROR LOADING IMAGE:\n"
                f"{image_path}\n"
                f"{e}"
            )

            # =====================
            # FALLBACK IMAGE
            # =====================

            image = Image.new(

                "RGB",

                (
                    CFG.IMAGE_SIZE,
                    CFG.IMAGE_SIZE
                )
            )

        # =========================
        # TRANSFORM
        # =========================

        if self.transform:

            image = self.transform(
                image
            )

        # =========================
        # LABEL
        # =========================

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


# =====================================
# COLLECT IMAGE PATHS
# =====================================

def collect_image_paths(

    folder_path,

    label
):

    image_paths = []

    labels = []

    if not os.path.exists(
        folder_path
    ):

        raise FileNotFoundError(

            f"\nFOLDER NOT FOUND:\n"
            f"{folder_path}"
        )

    files = sorted(
        os.listdir(folder_path)
    )

    for file in files:

        if file.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        ):

            image_paths.append(

                os.path.join(
                    folder_path,
                    file
                )
            )

            labels.append(label)

    return image_paths, labels


# =====================================
# CREATE DATASET
# =====================================

def create_dataset(

    split="train"
):

    image_paths = []

    labels = []

    # =========================
    # NEGATIVE DIRECTORY
    # =========================

    negative_dir = os.path.join(

       CFG.FINAL_DATASET_DIR,

       split,

       "IDC_negative"
    )

    # =========================
    # POSITIVE DIRECTORY
    # =========================

    positive_dir = os.path.join(

       CFG.FINAL_DATASET_DIR,

       split,

        "IDC_positive"
    )   

    # =========================
    # NEGATIVE IMAGES
    # =========================

    neg_paths, neg_labels = (

        collect_image_paths(

            negative_dir,

            0
        )
    )

    # =========================
    # POSITIVE IMAGES
    # =========================

    pos_paths, pos_labels = (

        collect_image_paths(

            positive_dir,

            1
        )
    )

    # =========================
    # MERGE
    # =========================

    image_paths.extend(
        neg_paths
    )

    image_paths.extend(
        pos_paths
    )

    labels.extend(
        neg_labels
    )

    labels.extend(
        pos_labels
    )

    combined = list(

       zip(image_paths, labels)
    )

    random.shuffle(combined)

    image_paths, labels = zip(*combined)

    image_paths = list(image_paths)

    labels = list(labels)

    # =========================
    # TRANSFORM
    # =========================

    if split == "train":

        transform = train_transform

    else:

        transform = val_transform

    # =========================
    # DATASET
    # =========================

    dataset = BreastCancerDataset(

        image_paths=image_paths,

        labels=labels,

        transform=transform
    )

    print(
        f"\n{split.upper()} DATASET:"
    )

    print(
        f"TOTAL IMAGES: "
        f"{len(dataset)}"
    )

    print(
        f"NEGATIVE: "
        f"{len(neg_paths)}"
    )

    print(
        f"POSITIVE: "
        f"{len(pos_paths)}"
    )

    return dataset


# =====================================
# CREATE DATALOADER
# =====================================

def create_dataloader(

    split="train"
):

    dataset = create_dataset(
        split
    )

    dataloader = DataLoader(

        dataset,

        batch_size=CFG.BATCH_SIZE,

        shuffle=(

            CFG.SHUFFLE_TRAIN

            if split == "train"

            else CFG.SHUFFLE_VAL
        ),

        num_workers=CFG.NUM_WORKERS,

        pin_memory=CFG.PIN_MEMORY,

        # =====================
        # FIX BATCHNORM ERROR
        # =====================

        drop_last=CFG.DROP_LAST,

)

    print(
        f"{split.upper()} "
        f"DATALOADER READY"
    )

    print(
        f"BATCHES: "
        f"{len(dataloader)}"
    )

    return dataloader


# =====================================
# TEST MODULE
# =====================================

if __name__ == "__main__":

    train_loader = create_dataloader(
        "train"
    )

    val_loader = create_dataloader(
        "val"
    )

    test_loader = create_dataloader(
        "test"
    )

    images, labels = next(
        iter(train_loader)
    )

    print("\nBATCH SHAPE:")

    print(images.shape)

    print(labels.shape)


