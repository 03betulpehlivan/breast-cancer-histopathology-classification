# =====================================
# IMPROVED SVM TRAINING PIPELINE
# =====================================

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

from sklearn.svm import SVC

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    classification_report,

    confusion_matrix,

    roc_curve
)

from configs.config import CFG

from dataset.dataset_loader import (
    create_dataloader
)

from models.feature_fusion import (
    FeatureFusionModel
)


# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(

    model,

    dataloader
):

    model.eval()

    features_list = []

    labels_list = []

    print("\nEXTRACTING FEATURES...\n")

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader
        ):

            images = images.to(
                CFG.DEVICE
            )

            # =========================
            # FEATURE EXTRACTION
            # =========================

            with torch.autocast(
                device_type="cuda",
                enabled=CFG.USE_AMP
            ):

                features = (
                    model.extract_features(
                        images
                    )
                )

            # =========================
            # TO NUMPY
            # =========================

            features = (
                features
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .cpu()
                .numpy()
            )

            features_list.append(
                features
            )

            labels_list.append(
                labels
            )

    # =================================
    # CONCATENATE
    # =================================

    features_array = np.concatenate(

        features_list,

        axis=0
    )

    features_array = features_array.astype(
        np.float32
    )

    labels_array = np.concatenate(

        labels_list,

        axis=0
    )

    print(
        f"\nFEATURE SHAPE: "
        f"{features_array.shape}"
    )

    return (

        features_array,

        labels_array
    )


# =====================================
# TRAIN SVM
# =====================================

def train_svm():

    print("\n========================")
    print("SVM TRAINING STARTED")
    print("========================")

    # =================================
    # DATALOADERS
    # =================================

    print("\nLOADING DATALOADERS...\n")

    train_loader = create_dataloader(
        split="train"
    )

    test_loader = create_dataloader(
        split="test"
    )

    print("\nDATALOADERS READY")

    # =================================
    # LOAD FEATURE MODEL
    # =================================

    print("\nLOADING FEATURE MODEL...\n")

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

    print("FEATURE MODEL READY")

    # =================================
    # TRAIN FEATURES
    # =================================

    X_train, y_train = (

        extract_features(

            model,

            train_loader
        )
    )

    # =================================
    # TEST FEATURES
    # =================================

    X_test, y_test = (

        extract_features(

            model,

            test_loader
        )
    )

    # =================================
    # SVM PIPELINE
    # =================================

    print("\nTRAINING SVM...\n")

    svm_pipeline = Pipeline([

        (
            "scaler",

            StandardScaler()
        ),

        (
            "svm",

            SVC(

                kernel="rbf",

                probability=True,

                C=2.0,

                gamma="auto",

                class_weight="balanced",

                random_state=CFG.SEED
            )
        )
    ])

    # =================================
    # TRAIN
    # =================================

    svm_pipeline.fit(

        X_train,

        y_train
    )

    print("SVM TRAINING COMPLETE")

    # =================================
    # SAVE MODEL
    # =================================

    svm_path = os.path.join(

        CFG.CHECKPOINT_DIR,

        "svm_model.pkl"
    )

    joblib.dump(

        svm_pipeline,

        svm_path
    )

    print(
        f"\nSVM SAVED:\n"
        f"{svm_path}"
    )

    # =================================
    # PREDICTIONS
    # =================================

    predictions = svm_pipeline.predict(
        X_test
    )

    probabilities = (

        svm_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # =================================
    # METRICS
    # =================================

    accuracy = accuracy_score(

        y_test,

        predictions
    )

    precision = precision_score(

        y_test,

        predictions
    )

    recall = recall_score(

        y_test,

        predictions
    )

    f1 = f1_score(

        y_test,

        predictions
    )

    roc_auc = roc_auc_score(

        y_test,

        probabilities
    )

    balanced_acc = balanced_accuracy_score(

        y_test,

        predictions
    )

    # =================================
    # RESULTS
    # =================================

    print("\n========================")
    print("SVM RESULTS")
    print("========================")

    print(
        f"\nAccuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_acc:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-Score  : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

    # =================================
    # CLASSIFICATION REPORT
    # =================================

    print("\n========================")
    print("CLASSIFICATION REPORT")
    print("========================\n")

    report = classification_report(

        y_test,

        predictions,

        target_names=CFG.CLASS_NAMES
    )

    print(report)

    # =================================
    # SAVE REPORT
    # =================================

    report_path = os.path.join(

        CFG.RESULT_DIR,

        "svm_classification_report.txt"
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

        y_test,

        predictions
    )

    plt.figure(
        figsize=(7, 6)

    )
    plt.style.use("ggplot")
    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=CFG.CLASS_NAMES,

        yticklabels=CFG.CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.title(
        "SVM Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(

        CFG.FIGURE_DIR,

        "svm_confusion_matrix.png"
    )

    plt.savefig(

        confusion_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # =================================
    # ROC CURVE
    # =================================

    fpr, tpr, thresholds = roc_curve(

        y_test,

        probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )
    plt.style.use("ggplot")
    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=(
            f"AUC = "
            f"{roc_auc:.4f}"
        )
    )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "SVM ROC Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    roc_path = os.path.join(

        CFG.FIGURE_DIR,

        "svm_roc_curve.png"
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

    summary_path = os.path.join(

        CFG.RESULT_DIR,

        "svm_results.txt"
    )

    with open(summary_path, "w") as f:

        f.write(
            "========== SVM RESULTS ==========\n\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy : "
            f"{balanced_acc:.4f}\n"
        )

        f.write(
            f"Precision : {precision:.4f}\n"
        )

        f.write(
            f"Recall : {recall:.4f}\n"
        )

        f.write(
            f"F1-Score : {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC : {roc_auc:.4f}\n"
        )

    print(
        f"\nSVM RESULTS SAVED:\n"
        f"{summary_path}"
    )

    # =================================
    # FINAL MESSAGE
    # =================================

    print("\n========================")
    print("SVM PIPELINE COMPLETE")
    print("========================")

    return svm_pipeline


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    train_svm()