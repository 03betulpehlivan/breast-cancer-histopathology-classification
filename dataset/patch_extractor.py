# =====================================
# IMPROVED PATCH EXTRACTION SYSTEM
# =====================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from configs.config import CFG


# =====================================
# READ IMAGE
# =====================================

def read_image(image_path):

    image = cv2.imread(
        image_path
    )

    # =========================
    # CHECK IMAGE
    # =========================

    if image is None:

        raise ValueError(

            f"\nIMAGE NOT FOUND:\n"
            f"{image_path}"
        )

    # =========================
    # BGR -> RGB
    # =========================

    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB
    )

    # =========================
    # RESIZE
    # =========================

    image = cv2.resize(

        image,

        (
            CFG.IMAGE_SIZE,
            CFG.IMAGE_SIZE
        ),

        interpolation=cv2.INTER_AREA
    )

    return image


# =====================================
# EXTRACT PATCHES
# =====================================

def extract_patches(

    image,

    return_coordinates=False
):

    patches = []

    coordinates = []

    patch_size = CFG.PATCH_SIZE

    image_size = CFG.IMAGE_SIZE

    # =========================
    # PATCH EXTRACTION
    # =========================

    for y in range(

        0,

        image_size,

        patch_size
    ):

        for x in range(

            0,

            image_size,

            patch_size
        ):

            patch = image[

                y:y + patch_size,

                x:x + patch_size
            ]

            # =====================
            # PATCH VALIDATION
            # =====================

            if patch.shape[0] != patch_size:

                continue

            if patch.shape[1] != patch_size:

                continue


            # =====================
            # REMOVE EMPTY PATCHES
            # =====================

            if np.mean(patch) > 240:

                  continue

            patches.append(
                patch
            )

            coordinates.append(
                (x, y)
            )

    # =========================
    # PATCH COUNT CHECK
    # =========================

    if len(patches) != CFG.NUM_PATCHES:

        print(
            f"\nWARNING:\n"
            f"Expected patches: "
            f"{CFG.NUM_PATCHES}\n"
            f"Found patches: "
            f"{len(patches)}"
        )

    # =========================
    # RETURN
    # =========================

    if return_coordinates:

        return patches, coordinates

    return patches


# =====================================
# DRAW PATCH GRID
# =====================================

def draw_patch_grid(image):

    image_copy = image.copy()

    patch_size = CFG.PATCH_SIZE

    # =========================
    # DRAW GRID
    # =========================

    for y in range(

        0,

        CFG.IMAGE_SIZE,

        patch_size
    ):

        cv2.line(

            image_copy,

            (0, y),

            (CFG.IMAGE_SIZE, y),

            (255, 0, 0),

            1
        )

    for x in range(

        0,

        CFG.IMAGE_SIZE,

        patch_size
    ):

        cv2.line(

            image_copy,

            (x, 0),

            (x, CFG.IMAGE_SIZE),

            (255, 0, 0),

            1
        )

    return image_copy


# =====================================
# VISUALIZE PATCHES
# =====================================

def visualize_patches(

    image,

    patches
):

    # =========================
    # GRID IMAGE
    # =========================

    grid_image = draw_patch_grid(
        image
    )

    plt.figure(
        figsize=(14, 14)
    )

    # =========================
    # ORIGINAL + GRID
    # =========================

    plt.subplot(
        5,
        4,
        1
    )

    plt.imshow(grid_image)

    plt.title(
        "Patch Grid",
        fontsize=12
    )

    plt.axis("off")

    # =========================
    # PATCHES
    # =========================

    for idx, patch in enumerate(patches):

        plt.subplot(
            5,
            4,
            idx + 2
        )

        plt.imshow(patch)

        plt.title(
            f"Patch {idx + 1}",
            fontsize=10
        )

        plt.axis("off")

    plt.tight_layout()

    os.makedirs(

       "figures/patches",

       exist_ok=True
    )

    plt.savefig(

       "figures/patches/patch_visualization.png",

       dpi=300,

       bbox_inches="tight"
    )

    plt.show()


# =====================================
# PATCH QUALITY CHECK
# =====================================

def check_patch_statistics(

    patches
):

    print("\nPATCH STATISTICS\n")

    print(
        f"Total patches: "
        f"{len(patches)}"
    )

    if len(patches) > 0:

       print(
           f"Patch shape: "
           f"{patches[0].shape}"
       )

    means = []

    stds = []

    for patch in patches:

        means.append(
            np.mean(patch)
        )

        stds.append(
            np.std(patch)
        )

    print(
        f"Mean intensity: "
        f"{np.mean(means):.2f}"
    )

    print(
        f"Std intensity: "
        f"{np.mean(stds):.2f}"
    )


# =====================================
# TEST PATCH PIPELINE
# =====================================

def test_patch_pipeline(

    image_path
):

    # =========================
    # READ IMAGE
    # =========================

    image = read_image(
        image_path
    )

    # =========================
    # EXTRACT PATCHES
    # =========================

    patches = extract_patches(
        image
    )

    # =========================
    # INFO
    # =========================

    check_patch_statistics(
        patches
    )

    # =========================
    # VISUALIZATION
    # =========================

    visualize_patches(

        image,

        patches
    )

    return patches


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    test_image = "test_image.png"

    if not os.path.exists(
        test_image
    ):

        print(
            "\nTEST IMAGE NOT FOUND"
        )

    else:

        test_patch_pipeline(
            test_image
        )


print("\nPATCH EXTRACTION MODULE READY")