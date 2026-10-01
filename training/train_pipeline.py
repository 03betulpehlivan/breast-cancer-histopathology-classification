# =====================================
# IMPROVED TRAINING PIPELINE v2
# =====================================

import csv
import time

import torch
import torch.nn as nn

from torch.amp import (
    autocast,
    GradScaler
)

from tqdm import tqdm

from configs.config import CFG

from dataset.dataset_loader import (
    create_dataloader
)

from models.feature_fusion import (
    FeatureFusionModel
)

from evaluation.visualization import (
    plot_training_history
)


# =====================================
# TRAIN ONE EPOCH
# =====================================

def train_one_epoch(

    model,

    dataloader,

    criterion,

    optimizer,

    scaler
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        dataloader
    )

    for images, labels in progress_bar:

        images = images.to(
            CFG.DEVICE
        )

        labels = labels.to(
            CFG.DEVICE
        )

        labels = labels.float().unsqueeze(1)

        # =========================
        # ZERO GRAD
        # =========================

        optimizer.zero_grad(

          set_to_none=True
        )

        # =========================
        # MIXED PRECISION
        # =========================

        with autocast(
            device_type="cuda",
            enabled=CFG.USE_AMP
        ):    

            outputs = model(
                images
            )

            loss = criterion(

                outputs,

                labels
            )

        # =========================
        # BACKWARD
        # =========================

        scaler.scale(loss).backward()

        # =========================
        # GRADIENT CLIPPING
        # =========================

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            CFG.GRADIENT_CLIP
        )

        scaler.step(optimizer)

        scaler.update()

        # =========================
        # STATS
        # =========================

        running_loss += loss.item()

        probabilities = torch.sigmoid(
           outputs
        )

        predicted = (
            probabilities > 0.5
        ).float()

        total += labels.size(0)

        correct += (

            predicted == labels

        ).sum().item()

        accuracy = (
            100 * correct / total
        )

        progress_bar.set_description(

            f"Loss: {loss.item():.4f} | "

            f"Acc: {accuracy:.2f}%"
        )

    epoch_loss = (

        running_loss /

        len(dataloader)
    )

    epoch_acc = (
        100 * correct / total
    )

    return epoch_loss, epoch_acc


# =====================================
# VALIDATION
# =====================================

def validate(

    model,

    dataloader,

    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                CFG.DEVICE
            )

            labels = labels.to(
                CFG.DEVICE
            )

            labels = labels.float().unsqueeze(1)

            # =====================
            # MIXED PRECISION
            # =====================

            with autocast(
               device_type="cuda",
               enabled=CFG.USE_AMP
            ):           

                outputs = model(
                    images
                )

                loss = criterion(

                    outputs,

                    labels
                )

            # =====================
            # STATS
            # =====================

            running_loss += loss.item()

            probabilities = torch.sigmoid(
               outputs
            ) 

            predicted = (
                 probabilities > 0.5
            ).float()

            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()

    val_loss = (

        running_loss /

        len(dataloader)
    )

    val_acc = (
        100 * correct / total
    )

    return val_loss, val_acc


# =====================================
# SAVE HISTORY CSV
# =====================================

def save_history_csv(history):

    with open(

        CFG.TRAIN_LOG_PATH,

        mode="w",

        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "epoch",

            "train_loss",

            "val_loss",

            "train_acc",

            "val_acc",

            "learning_rate"
        ])

        for i in range(

            len(history["train_loss"])
        ):

            writer.writerow([

                i + 1,

                history["train_loss"][i],

                history["val_loss"][i],

                history["train_acc"][i],

                history["val_acc"][i],

                history["learning_rate"][i]
            ])

    print(
        f"\nTRAIN LOG SAVED:\n"
        f"{CFG.TRAIN_LOG_PATH}"
    )


# =====================================
# TRAINING FUNCTION
# =====================================

def train_model():

    print("\n========================")
    print("TRAINING STARTED")
    print("========================")

    # =================================
    # DATALOADERS
    # =================================

    print("\nLOADING DATALOADERS...\n")

    train_loader = create_dataloader(
        split="train"
    )

    val_loader = create_dataloader(
        split="val"
    )

    print("\nDATALOADERS READY")

    # =================================
    # MODEL
    # =================================

    print("\nLOADING MODEL...\n")

    model = FeatureFusionModel()

    model = model.to(
        CFG.DEVICE
    )
    
    torch.backends.cudnn.benchmark = True
    
    
    print("MODEL READY")

    # =================================
    # LOSS FUNCTION
    # =================================

    criterion = nn.BCEWithLogitsLoss()

    # =================================
    # OPTIMIZER
    # =================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=CFG.LEARNING_RATE,

        weight_decay=CFG.WEIGHT_DECAY
    )

    # =================================
    # SCHEDULER
    # =================================

    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=CFG.SCHEDULER_FACTOR,

            patience=CFG.SCHEDULER_PATIENCE
        )

        if CFG.USE_SCHEDULER

        else None
    )

    # =================================
    # AMP SCALER
    # =================================

    scaler = GradScaler(
       "cuda",
       enabled=CFG.USE_AMP
    )

    # =================================
    # HISTORY
    # =================================

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_acc": [],

        "val_acc": [],

        "learning_rate": []
    }

    # =================================
    # EARLY STOPPING
    # =================================

    best_acc = 0.0

    best_loss = float("inf")

    patience_counter = 0

    # =================================
    # TRAIN LOOP
    # =================================

    for epoch in range(
        CFG.EPOCHS
    ):

        start_time = time.time()
        # =============================
        # BACKBONE UNFREEZE
        # =============================

        if epoch == CFG.FREEZE_EPOCHS :

            print(
                "\nUNFREEZING BACKBONES..."
            )

            model.unfreeze_backbones()
        print(

            f"\n========== "

            f"EPOCH {epoch+1}"

            f"/{CFG.EPOCHS} "

            f"=========="
        )

        # =================================
        # TRAIN
        # =================================

        train_loss, train_acc = (

            train_one_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scaler
            )
        )

        # =================================
        # VALIDATION
        # =================================

        val_loss, val_acc = (

            validate(

                model,

                val_loader,

                criterion
            )
        )

        # =================================
        # LEARNING RATE
        # =================================

        current_lr = optimizer.param_groups[0]["lr"]

        # =================================
        # SCHEDULER STEP
        # =================================

        if scheduler is not None:

            scheduler.step(val_loss)

        # =================================
        # HISTORY UPDATE
        # =================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_acc
        )

        history["learning_rate"].append(
            current_lr
        )

        # =================================
        # RESULTS
        # =================================

        elapsed = (
            time.time() - start_time
        )

        print(

            f"\nTRAIN LOSS : "
            f"{train_loss:.4f}"
        )

        print(

            f"TRAIN ACC  : "
            f"{train_acc:.2f}%"
        )

        print(

            f"VAL LOSS   : "
            f"{val_loss:.4f}"
        )

        print(

            f"VAL ACC    : "
            f"{val_acc:.2f}%"
        )

        print(

            f"LR         : "
            f"{current_lr:.6f}"
        )

        print(

            f"TIME       : "
            f"{elapsed:.2f}s"
        )

        # =================================
        # SAVE BEST MODEL
        # =================================

        if val_acc > best_acc:

            best_acc = val_acc
           
            best_loss = val_loss
           
            patience_counter = 0

            torch.save(

                model.state_dict(),

                CFG.BEST_MODEL_PATH
            )

            print(
                "\nBEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            print(

                f"\nNO IMPROVEMENT "

                f"({patience_counter}/"

                f"{CFG.EARLY_STOPPING_PATIENCE})"
            )

        # =================================
        # SAVE LAST MODEL
        # =================================

        torch.save(

            model.state_dict(),

            CFG.LAST_MODEL_PATH
        )

        # =================================
        # CUDA CACHE CLEAN
        # =================================

        if torch.cuda.is_available():

             torch.cuda.empty_cache()
             torch.cuda.synchronize()

        # =================================
        # EARLY STOPPING
        # =================================

        if (

            patience_counter >=

            CFG.EARLY_STOPPING_PATIENCE
        ):

            print("\nEARLY STOPPING TRIGGERED")

            break

    # =================================
    # SAVE HISTORY
    # =================================

    save_history_csv(history)

    # =================================
    # VISUALIZATION
    # =================================

    plot_training_history(history)

    # =================================
    # FINAL RESULTS
    # =================================

    print("\n========================")
    with open(

         CFG.FINAL_RESULTS_PATH,

         "w"
    ) as f:

        f.write(
            "========== FINAL TRAINING RESULTS ==========\n\n"
        )

        f.write(
            f"Best Validation Accuracy : "
            f"{best_acc:.2f}%\n"
        )

        f.write(
            f"Best Validation Loss : "
            f"{best_loss:.4f}\n"
        )
    print("TRAINING COMPLETE")
    print("========================")

    print(
        f"\nBEST VALIDATION ACC: "
        f"{best_acc:.2f}%"
    )

    print(
        f"BEST VALIDATION LOSS: "
        f"{best_loss:.4f}"
    )

    print(
        f"\nBEST MODEL PATH:\n"
        f"{CFG.BEST_MODEL_PATH}"
    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    train_model()