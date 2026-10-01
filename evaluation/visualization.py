# =====================================
# IMPROVED TRAINING VISUALIZATION
# =====================================

import os
import matplotlib.pyplot as plt

from configs.config import CFG


# =====================================
# PLOT TRAINING HISTORY
# =====================================

def plot_training_history(history):

    print("\nGENERATING TRAINING PLOTS...\n")

    # =================================
    # EXTRACT VALUES
    # =================================

    train_losses = history[
        "train_loss"
    ]

    val_losses = history[
        "val_loss"
    ]

    train_accs = history[
        "train_acc"
    ]

    val_accs = history[
        "val_acc"
    ]

    # =================================
    # OPTIONAL LEARNING RATE
    # =================================

    learning_rates = history.get(
        "learning_rate",
        None
    )

    # =================================
    # EPOCHS
    # =================================

    epochs = range(

        1,

        len(train_losses) + 1
    )

    # =================================
    # LOSS CURVE
    # =================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(

        epochs,

        train_losses,

        marker="o",

        linewidth=2,

        label="Train Loss",
       
       color="royalblue",
    )

    plt.plot(

        epochs,

        val_losses,

        marker="o",

        linewidth=2,

        label="Validation Loss",
        
        color="darkorange",
    )

    # =================================
    # BEST EPOCH
    # =================================

    best_epoch = (
        val_losses.index(
            min(val_losses)
        ) + 1
    )

    plt.axvline(

        best_epoch,

        linestyle="--",

        linewidth=1.5,

        label=(
            f"Best Epoch: "
            f"{best_epoch}"
        )
    )

    plt.xlabel(
        "Epoch",
        fontsize=12
    )

    plt.ylabel(
        "Loss",
        fontsize=12
    )

    plt.title(
        "Fusion Model Training and Validation Loss",
        fontsize=14
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # =================================
    # SAVE LOSS FIGURE
    # =================================

    loss_path = os.path.join(

        CFG.FIGURE_DIR,

        "loss_curve.png"
    )

    plt.savefig(

        loss_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"LOSS CURVE SAVED:\n"
        f"{loss_path}"
    )

    # =================================
    # ACCURACY CURVE
    # =================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(

        epochs,

        train_accs,

        marker="o",

        linewidth=2,

        label="Train Accuracy",

        color="royalblue",
    )

    plt.plot(

        epochs,

        val_accs,

        marker="o",

        linewidth=2,

        label="Validation Accuracy",

        color="darkorange",
    )

    # =================================
    # BEST EPOCH
    # =================================

    best_acc_epoch = (
        val_accs.index(
            max(val_accs)
        ) + 1
    )

    plt.axvline(

        best_acc_epoch,

        linestyle="--",

        linewidth=1.5,

        color="red",

        label=(
            f"Best Epoch: "
            f"{best_acc_epoch}"
        )
    )

    plt.xlabel(
        "Epoch",
        fontsize=12
    )

    plt.ylabel(
        "Accuracy",
        fontsize=12
    )

    plt.title(
        "Fusion Model Training and Validation Accuracy",
        fontsize=14
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # =================================
    # SAVE ACCURACY FIGURE
    # =================================

    acc_path = os.path.join(

        CFG.FIGURE_DIR,

        "accuracy_curve.png"
    )

    plt.savefig(

        acc_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"ACCURACY CURVE SAVED:\n"
        f"{acc_path}"
    )

    # =================================
    # LEARNING RATE CURVE
    # =================================

    if learning_rates is not None:

        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(

            epochs,

            learning_rates,

            marker="o",

            linewidth=2,

            color="purple",
        )

        plt.xlabel(
            "Epoch",
            fontsize=12
        )

        plt.ylabel(
            "Learning Rate",
            fontsize=12
        )

        plt.title(
            "Fusion Model Learning Rate Schedule",
            fontsize=14
        )

        plt.grid(True)

        plt.tight_layout()

        # =============================
        # SAVE LR FIGURE
        # =============================

        lr_path = os.path.join(

            CFG.FIGURE_DIR,

            "learning_rate_curve.png"
        )

        plt.savefig(

            lr_path,

            dpi=300,

            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"LEARNING RATE CURVE SAVED:\n"
            f"{lr_path}"
        )

    # =================================
    # TRAINING SUMMARY TXT
    # =================================

    summary_path = os.path.join(

        CFG.RESULT_DIR,

        "training_summary.txt"
    )

    with open(summary_path, "w") as f:

        f.write(
            "========== TRAINING SUMMARY ==========\n\n"
        )

        f.write(
            f"Best Validation Loss : "
            f"{min(val_losses):.4f}\n"
        )

        f.write(
            f"Best Validation Accuracy : "
            f"{max(val_accs):.4f}\n"
        )

        f.write(
            f"Best Epoch : "
            f"{best_acc_epoch}\n"
        )

    print(
        f"\nTRAINING SUMMARY SAVED:\n"
        f"{summary_path}"
    )

    # =================================
    # FINAL SUMMARY
    # =================================

    print("\n========================")
    print("TRAINING SUMMARY")
    print("========================")

    print(
        f"\nBest Validation Loss : "
        f"{min(val_losses):.4f}"
    )

    print(
        f"Best Validation Acc  : "
        f"{max(val_accs):.4f}"
    )

    print(
        f"Best Epoch           : "
        f"{best_acc_epoch}"
    )

    print("\nVISUALIZATION COMPLETED")



