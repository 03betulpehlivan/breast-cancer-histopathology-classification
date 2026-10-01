# =====================================
# SMART DATASET BUILDER
# ZIP -> TRAIN / VAL / TEST
# =====================================

import os
import zipfile
import random
import shutil

from tqdm import tqdm

from configs.config import CFG


# =====================================
# SETTINGS
# =====================================

ZIP_PATH = (
    "breast-histopathology-images.zip"
)

OUTPUT_DIR = (
    CFG.FINAL_DATASET_DIR
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# =====================================
# CREATE DIRECTORIES
# =====================================

def create_directories():

    # =============================
    # DATASET EXISTS CHECK
    # =============================

    if os.path.exists(OUTPUT_DIR):

         print("\nREMOVING OLD DATASET...\n")
 
         shutil.rmtree(OUTPUT_DIR)

    # =============================
    # SPLITS
    # =============================

    splits = [

        "train",

        "val",

        "test"
    ]

    # =============================
    # CLASSES
    # =============================

    classes = [

        "IDC_negative",

        "IDC_positive"
    ]

    # =============================
    # CREATE FOLDERS
    # =============================

    for split in splits:

        for cls in classes:

            path = os.path.join(

                OUTPUT_DIR,

                split,

                cls
            )

            os.makedirs(

                path,

                exist_ok=True
            )

    print("\nDIRECTORIES READY")


# =====================================
# EXTRACT ONLY NEEDED FILES
# =====================================

def extract_dataset():

    print("\nOPENING ZIP FILE...\n")

    if not os.path.exists(ZIP_PATH):

        raise FileNotFoundError(

            f"\nZIP FILE NOT FOUND:\n{ZIP_PATH}"
        )

    with zipfile.ZipFile(

        ZIP_PATH,

        "r"
    ) as zip_ref:

        file_list = zip_ref.namelist()

        print(
            f"TOTAL FILES: "
            f"{len(file_list)}"
        )

        # =============================
        # ONLY PNG FILES
        # =============================

        png_files = [

            f for f in file_list

            if f.lower().endswith(".png")
        ]

        print(
            f"PNG FILES: "
            f"{len(png_files)}"
        )

        # =============================
        # NEGATIVE / POSITIVE
        # =============================

        negative_files = []

        positive_files = []

        for file in png_files:

            parts = file.split("/")

            if len(parts) < 2:
                continue

            # =========================
            # CLASS LABEL
            # =========================

            if parts[-2] == "0":

                negative_files.append(
                    file
                )

            elif parts[-2] == "1":

                positive_files.append(
                    file
                )

        print(
            f"\nNEGATIVE: "
            f"{len(negative_files)}"
        )

        print(
            f"POSITIVE: "
            f"{len(positive_files)}"
        )

        # =============================
        # SHUFFLE
        # =============================

        random.shuffle(
            negative_files
        )

        random.shuffle(
            positive_files
        )

        # =============================
        # LIMIT DATASET
        # =============================

        neg_needed = (

            CFG.TRAIN_PER_CLASS +

            CFG.VAL_PER_CLASS +

            CFG.TEST_PER_CLASS
        )

        pos_needed = (

            CFG.TRAIN_PER_CLASS +

            CFG.VAL_PER_CLASS +

            CFG.TEST_PER_CLASS
        )

        negative_files = (
            negative_files[:neg_needed]
        )

        positive_files = (
            positive_files[:pos_needed]
        )

        # =============================
        # SPLIT FUNCTION
        # =============================

        def split_files(files):

            train_end = (
                CFG.TRAIN_PER_CLASS
            )

            val_end = (

                train_end +

                CFG.VAL_PER_CLASS
            )

            train_files = (
                files[:train_end]
            )

            val_files = (
                files[
                    train_end:val_end
                ]
            )

            test_files = (
                files[val_end:]
            )

            return (

                train_files,

                val_files,

                test_files
            )

        # =============================
        # SPLIT DATA
        # =============================

        neg_train, neg_val, neg_test = (
            split_files(negative_files)
        )

        pos_train, pos_val, pos_test = (
            split_files(positive_files)
        )

        # =============================
        # COPY FUNCTION
        # =============================

        def extract_files(

            files,

            split,

            class_name
        ):

            save_dir = os.path.join(

                OUTPUT_DIR,

                split,

                class_name
            )

            for file in tqdm(
                files,
                desc=f"{split}-{class_name}"
            ):

                filename = os.path.basename(
                    file
                )

                destination = os.path.join(

                    save_dir,

                    filename
                )

                with zip_ref.open(file) as source:

                    with open(

                        destination,

                        "wb"
                    ) as target:

                        shutil.copyfileobj(

                            source,

                            target
                        )

        # =============================
        # EXTRACT NEGATIVE
        # =============================

        print("\nEXTRACTING NEGATIVE...\n")

        extract_files(

            neg_train,

            "train",

            "IDC_negative"
        )

        extract_files(

            neg_val,

            "val",

            "IDC_negative"
        )

        extract_files(

            neg_test,

            "test",

            "IDC_negative"
        )

        # =============================
        # EXTRACT POSITIVE
        # =============================

        print("\nEXTRACTING POSITIVE...\n")

        extract_files(

            pos_train,

            "train",

            "IDC_positive"
        )

        extract_files(

            pos_val,

            "val",

            "IDC_positive"
        )

        extract_files(

            pos_test,

            "test",

            "IDC_positive"
        )

    print("\nDATASET CREATION COMPLETE")


# =====================================
# DATASET INFO
# =====================================

def dataset_info():

    print("\n========================")
    print("FINAL DATASET")
    print("========================")

    splits = [

        "train",

        "val",

        "test"
    ]

    classes = [

        "IDC_negative",

        "IDC_positive"
    ]

    for split in splits:

        print(f"\n{split.upper()}")

        for cls in classes:

            path = os.path.join(

                OUTPUT_DIR,

                split,

                cls
            )

            total = len(
                os.listdir(path)
            )

            print(
                f"{cls}: {total}"
            )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    create_directories()

    extract_dataset()

    dataset_info()

