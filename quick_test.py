# =====================================
# QUICK MODEL TEST
# =====================================

import torch

from configs.config import CFG

from models.feature_fusion import (
    FeatureFusionModel
)


# =====================================
# START
# =====================================

print("\n========================")
print("QUICK TEST STARTED")
print("========================")

print(
    f"\nDEVICE: {CFG.DEVICE}"
)


# =====================================
# LOAD MODEL
# =====================================

print("\nLOADING MODEL...\n")

model = FeatureFusionModel()

model = model.to(
    CFG.DEVICE
)

model.eval()

print("MODEL LOADED")


# =====================================
# MODEL INFO
# =====================================

total_params = sum(

    p.numel()

    for p in model.parameters()
)

trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)

print(
    f"\nTOTAL PARAMETERS: "
    f"{total_params:,}"
)

print(
    f"TRAINABLE PARAMETERS: "
    f"{trainable_params:,}"
)


# =====================================
# DUMMY INPUT
# =====================================

dummy_input = torch.randn(

    2,

    3,

    CFG.IMAGE_SIZE,

    CFG.IMAGE_SIZE

).to(CFG.DEVICE)

print(
    f"\nDUMMY INPUT CREATED:"
)

print(
    dummy_input.shape
)


# =====================================
# FORWARD PASS
# =====================================

print("\nRUNNING FORWARD PASS...\n")

with torch.no_grad():

    output = model(
        dummy_input
    )

print("FORWARD PASS COMPLETE")


# =====================================
# RESULTS
# =====================================

print("\n========================")
print("TEST RESULTS")
print("========================")

print(
    f"\nInput Shape: "
    f"{dummy_input.shape}"
)

print(
    f"Output Shape: "
    f"{output.shape}"
)

print(
    f"Output Device: "
    f"{output.device}"
)

print(
    f"\nOutput Tensor:\n"
)

print(output)


# =====================================
# CUDA MEMORY
# =====================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated() / 1024**2
    )

    reserved = (
        torch.cuda.memory_reserved() / 1024**2
    )

    print(
        f"\nCUDA Allocated: "
        f"{allocated:.2f} MB"
    )

    print(
        f"CUDA Reserved: "
        f"{reserved:.2f} MB"
    )


# =====================================
# SUCCESS
# =====================================

print("\n========================")
print("QUICK TEST SUCCESSFUL")
print("========================")