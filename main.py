# =====================================
# IMPROVED MAIN PIPELINE
# =====================================

import os
import torch
import time
from configs.config import CFG

from utils.seed_utils import (
    set_seed
)

from training.train_pipeline import (
    train_model
)

from evaluation.metrics import (
    evaluate_model
)

from training.train_svm import (
    train_svm
)


# =====================================
# SYSTEM INFO
# =====================================

def print_system_info():

    print("\n========================")
    print("SYSTEM INFORMATION")
    print("========================")
    
    print(
        f"\nPROJECT: "
        f"{CFG.PROJECT_NAME}"
    )

    print(
        f"EXPERIMENT: "
        f"{CFG.EXPERIMENT_NAME}"
    )

    print(
        f"DEVICE: "
        f"{CFG.DEVICE}"
    )

    print(
        f"IMAGE SIZE: "
        f"{CFG.IMAGE_SIZE}"
    )

    print(
        f"BATCH SIZE: "
        f"{CFG.BATCH_SIZE}"
    )

    print(
        f"EPOCHS: "
        f"{CFG.EPOCHS}"
    )

    # =================================
    # CUDA INFO
    # =================================

    if torch.cuda.is_available():

        print(
            f"\nGPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (

            torch.cuda.get_device_properties(0)

            .total_memory / 1024**3
        )

        print(
            f"GPU MEMORY: "
            f"{total_memory:.2f} GB"
        )

    print("========================")


# =====================================
# CHECK REQUIRED FILES
# =====================================

def check_project_structure():

    print("\nCHECKING PROJECT...\n")

    required_dirs = [

        CFG.TRAIN_DIR,

        CFG.VAL_DIR,

        CFG.TEST_DIR
    ]

    for directory in required_dirs:

        if not os.path.exists(
            directory
        ):

            raise FileNotFoundError(

                f"\nMISSING DIRECTORY:\n"
                f"{directory}"
            )

    print("PROJECT STRUCTURE OK")


# =====================================
# MAIN
# =====================================

def main():

    print("\n========================")
    print("BREAST CANCER AI PIPELINE")
    print("========================")
    pipeline_start = time.time()
    # =================================
    # SET SEED
    # =================================

    set_seed(CFG.SEED)

    # =================================
    # SYSTEM INFO
    # =================================

    print_system_info()

    # =================================
    # CHECK DATASET
    # =================================

    check_project_structure()

    # =================================
    # TRAIN MODEL
    # =================================

    print("\n========================")
    print("STEP 1 - TRAINING")
    print("========================")

    train_model()

    # =================================
    # CHECK MODEL
    # =================================

    if os.path.exists(

        CFG.BEST_MODEL_PATH
    ):

        print("\nBEST MODEL FOUND\n")

        # =============================
        # EVALUATION
        # =============================

        print("\n========================")
        print("STEP 2 - EVALUATION")
        print("========================")

        evaluate_model()

        # =============================
        # SVM
        # =============================

        print("\n========================")
        print("STEP 3 - SVM PIPELINE")
        print("========================")

        train_svm()

        # =============================
        # GRADCAM INFO
        # =============================

        print("\n========================")
        print("STEP 4 - GRADCAM")
        print("========================")

        print(
            "\nAFTER TRAINING YOU CAN RUN:\n"
        )

        print(
            "python -m evaluation.gradcam"
        )

    else:

        print(
            "\nBEST MODEL NOT FOUND\n"
        )

    # =================================
    # FINAL MESSAGE
    # =================================

    print("\n========================")
    print("PIPELINE COMPLETE")
    print("========================")

    total_time = (

        time.time() -

        pipeline_start
    )

    print(
        f"\nTOTAL PIPELINE TIME : "
        f"{total_time/60:.2f} MINUTES"
    )

    print(
        f"\nBEST MODEL:\n"
        f"{CFG.BEST_MODEL_PATH}"
    )

    print(
        f"\nRESULTS DIRECTORY:\n"
        f"{CFG.RESULT_DIR}"
    )

    print(
        f"\nFIGURES DIRECTORY:\n"
        f"{CFG.FIGURE_DIR}"
    )

    summary_path = os.path.join(

        CFG.RESULT_DIR,

        "pipeline_summary.txt"
    )

    with open(summary_path, "w") as f:

        f.write(
            "========== PIPELINE SUMMARY ==========\n\n"
        )

        f.write(
            f"Best Model : "
            f"{CFG.BEST_MODEL_PATH}\n"
        )

        f.write(
            f"Results Directory : "
            f"{CFG.RESULT_DIR}\n"
        )

        f.write(
            f"Figures Directory : "
            f"{CFG.FIGURE_DIR}\n"
        )

    print(
        f"\nPIPELINE SUMMARY SAVED:\n"
        f"{summary_path}"
    )
# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    main()