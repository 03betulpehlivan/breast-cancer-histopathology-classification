# =====================================
# IMPROVED SEED UTILITIES
# =====================================

import os
import random
import numpy as np
import torch


# =====================================
# SET GLOBAL SEED
# =====================================

def set_seed(seed=42):

    print("\nSETTING RANDOM SEED...\n")

    # =================================
    # PYTHON
    # =================================

    random.seed(seed)

    # =================================
    # NUMPY
    # =================================

    np.random.seed(seed)

    # =================================
    # PYTORCH CPU
    # =================================

    torch.manual_seed(seed)

    # =================================
    # PYTORCH CUDA
    # =================================

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # =================================
    # PYTHON HASH SEED
    # =================================

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    # =================================
    # CUDNN SETTINGS
    # =================================

    torch.backends.cudnn.deterministic = False
    
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # =================================
    # INFO
    # =================================

    print(
        f"SEED SET TO: {seed}"
    )

    print(
        f"DETERMINISTIC MODE: TRUE"
    )

    print(
        f"BENCHMARK MODE: FALSE"
    )

    return seed


# =====================================
# WORKER INIT FUNCTION
# =====================================

def seed_worker(worker_id):

    worker_seed = (

        torch.initial_seed() % 2**32
    )

    np.random.seed(worker_seed)

    random.seed(worker_seed)


# =====================================
# GENERATOR
# =====================================

def get_generator(seed=42):

    generator = torch.Generator()

    generator.manual_seed(seed)

    return generator


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    set_seed(42)

    print("\nSEED UTILITIES READY")