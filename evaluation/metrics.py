# =====================================
# IMPROVED EVALUATION METRICS
# =====================================

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (


    classification_report,

    confusion_matrix,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    roc_curve,

    balanced_accuracy_score
)

from configs.config import CFG

from dataset.dataset_loader import (
    create_dataloader
)

from models.feature_fusion import (
    FeatureFusionModel
)


# =====================================
# EVALUATE MODEL
# =====================================

def evaluate_model():

    print("\n========================")
    print("EVALUATION STARTED")
    print("========================")

    # =================================
    # LOAD TEST DATA
    # =================================

    print("\nLOADING TEST DATA...\n")

    test_loader = create_dataloader(
        split="test"
    )

    print("TEST LOADER READY")

    # =================================
    # LOAD MODEL
    # =================================

    print("\nLOADING MODEL...\n")

    model = FeatureFusionModel()

    model.load_state_dict(

        torch.load(

            CFG.BEST_MODEL_PATH,

            map_location=CFG.DEVICE
        )
    )

    model = model.to(
        CFG.DEVICE
    )

    model.eval()

    print("MODEL LOADED")

    # =================================
    # STORAGE
    # =================================

    all_labels = []

    all_predictions = []

    all_probabilities = []

    # =================================
    # EVALUATION LOOP
    # =================================

    print("\nRUNNING INFERENCE...\n")

    with torch.no_grad():

        for images, labels in tqdm(
            test_loader
        ):

            images = images.to(
                CFG.DEVICE
            )

            labels = labels.to(
                CFG.DEVICE
            )

            # =========================
            # FORWARD
            # =========================

            outputs = model(
                images
            )

            # =========================
            # PROBABILITIES
            # =========================

            probabilities = torch.sigmoid(
               outputs
            ).squeeze()

            # =========================
            # PREDICTIONS
            # =========================

            predicted = (
              probabilities > 0.5
            ).long()
            # =========================
            # STORE
            # =========================

            all_labels.extend(

                labels.cpu().numpy()
            )

            all_predictions.extend(

                predicted.cpu().numpy()
            )

            all_probabilities.extend(

              probabilities
             .cpu()
             .numpy()
            )

    # =================================
    # NUMPY
    # =================================

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )

    np.save(

       os.path.join(

             CFG.RESULT_DIR,

            "all_labels.npy"
       ), 

       all_labels
    )

    np.save(

        os.path.join(

             CFG.RESULT_DIR,

            "all_probabilities.npy"
        ),

        all_probabilities
    )

    # =================================
    # METRICS
    # =================================

    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )

    precision = precision_score(

        all_labels,

        all_predictions
    )

    recall = recall_score(

        all_labels,

        all_predictions
    )

    f1 = f1_score(

        all_labels,

        all_predictions
    )

    roc_auc = roc_auc_score(

       all_labels,

       all_probabilities
    )

    balanced_acc = balanced_accuracy_score(

       all_labels,

       all_predictions
    )
  

    # =================================
    # PRINT RESULTS
    # =================================

    print("\n========================")
    print("TEST RESULTS")
    print("========================")

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_acc:.4f}"
    )

    # =================================
    # SAVE METRICS
    # =================================

    with open(

        CFG.METRICS_PATH,

        "w"
    ) as f:

        f.write(
            "========== TEST RESULTS ==========\n\n"
        )

        f.write(
            f"Accuracy  : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision : {precision:.4f}\n"
        )

        f.write(
            f"Recall    : {recall:.4f}\n"
        )

        f.write(
            f"F1-Score  : {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC   : {roc_auc:.4f}\n"
        )

        f.write(
             f"Balanced Accuracy : "
             f"{balanced_acc:.4f}\n"
        )

    print(
        f"\nMETRICS SAVED:\n"
        f"{CFG.METRICS_PATH}"
    )

    # =================================
    # CLASSIFICATION REPORT
    # =================================

    print("\n========================")
    print("CLASSIFICATION REPORT")
    print("========================\n")

    report = classification_report(

        all_labels,

        all_predictions,

        target_names=CFG.CLASS_NAMES
    )

    print(report)

    # =================================
    # SAVE REPORT
    # =================================

    report_path = os.path.join(

        CFG.RESULT_DIR,

        "classification_report.txt"
    )

    with open(

        report_path,

        "w"
    ) as f:

        f.write(report)

    # =================================
    # CONFUSION MATRIX
    # =================================

    cm = confusion_matrix(

        all_labels,

        all_predictions
    )

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=CFG.CLASS_NAMES,

        yticklabels=CFG.CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "True Label",
        fontsize=12
    )

    plt.title(
        "Fusion Model Confusion Matrix",
        fontsize=14
    )

    plt.tight_layout()

    # =================================
    # SAVE CONFUSION MATRIX
    # =================================

    confusion_path = os.path.join(

        CFG.FIGURE_DIR,

        "confusion_matrix.png"
    )

    plt.savefig(

        confusion_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"\nCONFUSION MATRIX SAVED:\n"
        f"{confusion_path}"
    )

    # =================================
    # ROC CURVE
    # =================================

    fpr, tpr, thresholds = roc_curve(

        all_labels,

        all_probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(

        fpr,

        tpr,

        linewidth=2,
        color="darkorange",

        label=(
            f"AUC = {roc_auc:.4f}"
        )
    )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12
    )

    plt.title(
        "Fusion Model ROC Curve",
        fontsize=14
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # =================================
    # SAVE ROC CURVE
    # =================================

    roc_path = os.path.join(

        CFG.FIGURE_DIR,

        "roc_curve.png"
    )

    plt.savefig(

        roc_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"\nROC CURVE SAVED:\n"
        f"{roc_path}"
    )

    # =================================
    # FINAL MESSAGE
    # =================================

    print("\n========================")
    print("EVALUATION COMPLETED")
    print("========================")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    evaluate_model()