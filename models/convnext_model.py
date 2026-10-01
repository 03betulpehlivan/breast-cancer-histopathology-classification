# =====================================
# IMPROVED CONVNEXT MODEL
# =====================================

import torch
import torch.nn as nn

import timm

from configs.config import CFG


# =====================================
# CONVNEXT MODEL
# =====================================

class ConvNeXtModel(nn.Module):

    def __init__(

        self,

        pretrained=True
    ):

        super().__init__()

        # =================================
        # LOAD BACKBONE
        # =================================

        self.backbone = timm.create_model(

            CFG.CONVNEXT_MODEL_NAME,

            pretrained=pretrained,

            num_classes=0,

            global_pool="avg"
        )

        # =================================
        # FEATURE DIMENSION
        # =================================

        self.feature_dim = (
            self.backbone.num_features
        )

        
        # =================================
        # CLASSIFIER HEAD
        # =================================

        self.classifier = nn.Sequential(

            # =============================
            # FC 1
            # =============================

            nn.Linear(

                self.feature_dim,

                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.GELU(),

            nn.Dropout(0.25),

            # =============================
            # FC 2
            # =============================

            nn.Linear(

                512,

                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            # =============================
            # OUTPUT
            # =============================

            nn.Linear(

                256,

                CFG.NUM_CLASSES
            )
        )

        # =================================
        # INITIALIZE CLASSIFIER
        # =================================

        self._initialize_weights()

    # =====================================
    # INITIALIZE WEIGHTS
    # =====================================

    def _initialize_weights(self):

        for module in self.classifier:

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0
                    )

    # =====================================
    # FREEZE BACKBONE
    # =====================================

    def freeze_backbone(self):

        print(
            "\nFREEZING BACKBONE..."
        )

        for param in self.backbone.parameters():

            param.requires_grad = False

    # =====================================
    # UNFREEZE BACKBONE
    # =====================================

    def unfreeze_backbone(self):

        print(
            "\nUNFREEZING BACKBONE..."
        )

        for param in self.backbone.parameters():

            param.requires_grad = True

    # =====================================
    # EXTRACT FEATURES
    # =====================================

    def extract_features(

        self,

        x
    ):

        features = self.backbone(
            x
        )
        features = features.float()

        return features

    # =====================================
    # FORWARD
    # =====================================

    def forward(

        self,

        x,

        return_features=False
    ):

        # =============================
        # FEATURE EXTRACTION
        # =============================

        features = self.extract_features(
            x
        )

        # =============================
        # CLASSIFICATION
        # =============================

        output = self.classifier(
            features
        )

        # =============================
        # OPTIONAL FEATURE RETURN
        # =============================

        if return_features:

            return output, features

        return output

    # =====================================
    # MODEL SUMMARY
    # =====================================

    def get_model_info(self):

        total_params = sum(

            p.numel()

            for p in self.parameters()
        )

        trainable_params = sum(

            p.numel()

            for p in self.parameters()

            if p.requires_grad
        )

        print("\n========================")
        print("CONVNEXT MODEL INFO")
        print("========================")

        print(
            f"\nBackbone: "
            f"{CFG.CONVNEXT_MODEL_NAME}"
        )

        print(
            f"Feature Dimension: "
            f"{self.feature_dim}"
        )

        print(
            f"Total Parameters: "
            f"{total_params:,}"
        )

        print(
            f"Trainable Parameters: "
            f"{trainable_params:,}"
        )

        print("========================")


# =====================================
# TEST MODEL
# =====================================

def test_model():

    print("\nTESTING CONVNEXT MODEL...\n")

    model = ConvNeXtModel()

    model = model.to(
        CFG.DEVICE
    )

    # =================================
    # MODEL INFO
    # =================================

    model.get_model_info()

    # =================================
    # DUMMY INPUT
    # =================================

    dummy_input = torch.randn(

        2,

        3,

        CFG.IMAGE_SIZE,

        CFG.IMAGE_SIZE

    ).to(CFG.DEVICE)

    # =================================
    # FORWARD
    # =================================

    output, features = model(

        dummy_input,

        return_features=True
    )

    print("\n========================")
    print("FORWARD TEST")
    print("========================")

    print(
        f"\nInput Shape: "
        f"{dummy_input.shape}"
    )

    print(
        f"Feature Shape: "
        f"{features.shape}"
    )

    print(
        f"Output Shape: "
        f"{output.shape}"
    )

    print("========================")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    test_model()


