# =====================================
# IMPROVED FEATURE FUSION SYSTEM
# =====================================

import torch
import torch.nn as nn

from configs.config import CFG

from models.vit_model import ViTModel
from models.swin_model import SwinModel
from models.convnext_model import ConvNeXtModel


# =====================================
# FEATURE FUSION MODEL
# =====================================

class FeatureFusionModel(nn.Module):

    def __init__(

        self,

        pretrained=True
    ):

        super().__init__()

        # =================================
        # BACKBONES
        # =================================

        print("\nLOADING BACKBONES...\n")

        self.vit_model = ViTModel(
            pretrained=pretrained
        )

        self.swin_model = SwinModel(
            pretrained=pretrained
        )

        self.convnext_model = ConvNeXtModel(
            pretrained=pretrained
        )

        print("BACKBONES READY")

        # =================================
        # FEATURE DIMENSIONS
        # =================================

        self.vit_dim = (
            self.vit_model.feature_dim
        )

        self.swin_dim = (
            self.swin_model.feature_dim
        )

        self.convnext_dim = (
            self.convnext_model.feature_dim
        )

        # =================================
        # TOTAL FEATURE DIM
        # =================================

        self.total_dim = (

            self.vit_dim +

            self.swin_dim +

            self.convnext_dim
        )

        # =================================
        # FEATURE NORMALIZATION
        # =================================

        self.feature_norm = nn.BatchNorm1d(
            self.total_dim
        )
        
        # =================================
        # FINAL CLASSIFIER
        # =================================

        self.classifier = nn.Sequential(

            # =============================
            # FC 1
            # =============================

            nn.Linear(

                self.total_dim,

                1024
            ),

            nn.BatchNorm1d(
                1024
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            # =============================
            # FC 2
            # =============================

            nn.Linear(

                1024,

                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.GELU(),

            nn.Dropout(0.15),

            # =============================
            # FC 3
            # =============================

            nn.Linear(

                512,

                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.GELU(),

            nn.Dropout(0.15),

            # =============================
            # OUTPUT
            # =============================

            nn.Linear(

                128,

                CFG.NUM_CLASSES
            )
        )

        # =================================
        # INITIALIZE WEIGHTS
        # =================================

        self._initialize_weights()

    # =====================================
    # INITIALIZE CLASSIFIER
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
    # FREEZE BACKBONES
    # =====================================

    def freeze_backbones(self):

        print(
            "\nFREEZING BACKBONES..."
        )

        for model in [

            self.vit_model,

            self.swin_model,

            self.convnext_model
        ]:

            for param in model.parameters():

                param.requires_grad = False

    # =====================================
    # UNFREEZE BACKBONES
    # =====================================

    def unfreeze_backbones(self):

        print(
            "\nUNFREEZING BACKBONES..."
        )

        for model in [

            self.vit_model,

            self.swin_model,

            self.convnext_model
        ]:

            for param in model.parameters():

                param.requires_grad = True

    # =====================================
    # EXTRACT FEATURES
    # =====================================

    def extract_features(

        self,

        x
    ):

        # =============================
        # ViT FEATURES
        # =============================

        vit_features = (

            self.vit_model.extract_features(
                x
            )
        )

        # =============================
        # SWIN FEATURES
        # =============================

        swin_features = (

            self.swin_model.extract_features(
                x
            )
        )

        # =============================
        # CONVNEXT FEATURES
        # =============================

        convnext_features = (

            self.convnext_model.extract_features(
                x
            )
        )

        # =============================
        # CONCATENATION
        # =============================

        fused_features = torch.cat(

            [

                vit_features,

                swin_features,

                convnext_features
            ],

            dim=1
        )

        # =============================
        # NORMALIZATION
        # =============================

        fused_features = self.feature_norm(
            fused_features
        )
        fused_features = fused_features.float()
        return fused_features

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

        fused_features = (

            self.extract_features(x)
        )

        # =============================
        # CLASSIFICATION
        # =============================

        output = self.classifier(
            fused_features
        )

        # =============================
        # OPTIONAL FEATURE RETURN
        # =============================

        if return_features:

            return output, fused_features

        return output

    # =====================================
    # MODEL INFO
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
        print("FEATURE FUSION INFO")
        print("========================")

        print(
            f"\nViT Feature Dim: "
            f"{self.vit_dim}"
        )

        print(
            f"Swin Feature Dim: "
            f"{self.swin_dim}"
        )

        print(
            f"ConvNeXt Feature Dim: "
            f"{self.convnext_dim}"
        )

        print(
            f"Total Feature Dim: "
            f"{self.total_dim}"
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

    print("\nTESTING FEATURE FUSION...\n")

    model = FeatureFusionModel()

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
        f"Fused Feature Shape: "
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

