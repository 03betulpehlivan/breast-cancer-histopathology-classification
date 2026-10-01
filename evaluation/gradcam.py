# =====================================
# IMPROVED GRADCAM VISUALIZATION
# =====================================

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

from PIL import Image
from torchvision import transforms

from configs.config import CFG

from models.feature_fusion import FeatureFusionModel
from models.vit_model import ViTModel
from models.swin_model import SwinModel
from models.convnext_model import ConvNeXtModel


# High-resolution output for heatmap overlays (presentation quality)
GRADCAM_OVERLAY_SIZE = CFG.IMAGE_SIZE
GRADCAM_GALLERY_DIR = "figures/gradcam_gallery"
GRADCAM_GALLERY_SAMPLES = 8


# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================
# LOAD IMAGE
# =====================================

def load_image(image_path, display_size=None):
    display_size = display_size or GRADCAM_OVERLAY_SIZE
    image = Image.open(image_path).convert("RGB")
    image_model = image.resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
    image_display = image.resize((display_size, display_size))
    original = np.array(image_display)
    input_tensor = transform(image_model).unsqueeze(0)
    return original, input_tensor


# =====================================
# TARGET LAYER RESOLUTION
# =====================================

def get_target_layer(model, backbone="fusion"):
    """
    Resolve a spatial GradCAM hook layer for ViT, Swin, ConvNeXt, or fusion.

    backbone: "vit" | "swin" | "convnext" | "fusion"
    For fusion, ConvNeXt final stage block is used (spatial features).
    """
    if isinstance(model, FeatureFusionModel):
        if backbone == "vit":
            bb = model.vit_model.backbone
            if hasattr(bb, "blocks"):
                return bb.blocks[-1].norm1
            raise ValueError("ViT backbone has no encoder blocks.")
        if backbone == "swin":
            bb = model.swin_model.backbone
            if hasattr(bb, "layers"):
                return bb.layers[-1].blocks[-1]
            raise ValueError("Swin backbone has no stage layers.")
        if backbone in ("convnext", "fusion"):
            bb = model.convnext_model.backbone
            if hasattr(bb, "stages"):
                return bb.stages[-2].blocks[-1]
            raise ValueError("ConvNeXt backbone has no stages.")
        raise ValueError(f"Unknown backbone for fusion model: {backbone}")

    if isinstance(model, ViTModel):
        bb = model.backbone
        if hasattr(bb, "blocks"):
            return bb.blocks[-1].norm1
        raise ValueError("ViT backbone has no encoder blocks.")

    if isinstance(model, SwinModel):
        bb = model.backbone
        if hasattr(bb, "layers"):
            return bb.layers[-1].blocks[-1]
        raise ValueError("Swin backbone has no stage layers.")

    if isinstance(model, ConvNeXtModel):
        bb = model.backbone
        if hasattr(bb, "stages"):
            return bb.stages[-2].blocks[-1]
        raise ValueError("ConvNeXt backbone has no stages.")

    raise TypeError(
        f"Unsupported model type for GradCAM: {type(model).__name__}"
    )


# =====================================
# GRADCAM
# =====================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._forward_handle = None
        self._backward_handle = None

        self._forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self._backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def remove_hooks(self):

        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def save_activation(self, module, input, output):

        print("FORWARD HOOK ACTIVATED")

        if isinstance(output, tuple):
            output = output[0]

        self.activations = output

        # ONLY retain grad if tensor supports gradients
        if (
            isinstance(self.activations, torch.Tensor)
            and self.activations.requires_grad
        ):
            self.activations.retain_grad()

    def save_gradient(self, module, grad_input, grad_output):

        print("BACKWARD HOOK ACTIVATED")

        if grad_output is not None and grad_output[0] is not None:
            self.gradients = grad_output[0]

    @staticmethod
    def _is_square_token_grid(num_tokens):

        side = int(np.sqrt(num_tokens))

        return side * side == num_tokens

    @staticmethod
    def _is_channels_first_spatial(tensor):
        """
        True for ConvNeXt-style (C, H, W) feature maps.
        """

        if tensor.ndim != 3:
            return False

        channels, height, width = tensor.shape

        return (
            height > 1
            and width > 1
            and channels >= height
            and channels >= width
        )

    @staticmethod
    def _is_hwc_spatial(tensor):
        """
        True for Swin-style (H, W, C) feature maps.
        """

        if tensor.ndim != 3:
            return False

        height, width, channels = tensor.shape

        return (
            height > 1
            and width > 1
            and channels >= 32
            and channels >= height
        )

    @staticmethod
    def _is_token_channel_map(tensor):
        """
        True for (tokens, channels) transformer layouts.
        """

        if tensor.ndim != 2:
            return False

        tokens, channels = tensor.shape

        return channels >= 32 and tokens != channels

    @staticmethod
    def _is_batch_token_channel_map(tensor):
        """
        True for (B, tokens, channels) transformer layouts.
        """

        if tensor.ndim != 3:
            return False

        batch, tokens, channels = tensor.shape

        if batch > 8:
            return False

        return (
            channels >= 32
            and tokens != channels
            and not GradCAM._is_channels_first_spatial(tensor[0])
        )

    def _squeeze_batch_dim(self, activations, gradients):
        """Remove leading batch dimension when both tensors share it."""
        if activations.ndim < 2 or gradients.ndim < 2:
            return activations, gradients

        if activations.shape[0] != gradients.shape[0]:
            return activations, gradients

        batch = activations.shape[0]
        if batch == 1:
            return activations[0], gradients[0]

        if batch <= 8 and (
            self._is_batch_token_channel_map(activations)
            or activations.ndim == 4
        ):
            return activations[0], gradients[0]

        return activations, gradients

    def _tokens_to_spatial(self, activations, gradients):
        """(tokens, channels) -> (channels, height, width)."""
        n_tokens = activations.shape[0]

        if not self._is_square_token_grid(n_tokens) and n_tokens > 1:
            activations = activations[1:]
            gradients = gradients[1:]
            n_tokens = activations.shape[0]

        side = int(np.sqrt(n_tokens))
        if self._is_square_token_grid(n_tokens):
            activations = activations.reshape(side, side, -1).transpose(2, 0, 1)
            gradients = gradients.reshape(side, side, -1).transpose(2, 0, 1)
            return activations, gradients

        activations = activations.T[:, np.newaxis, :]
        gradients = gradients.T[:, np.newaxis, :]
        return activations, gradients

    def _to_spatial_maps(self, activations, gradients):
        """Convert ViT/Swin token maps or Conv feature maps to (C, H, W)."""
        print("ACTIVATION SHAPE:", activations.shape)
        print("GRADIENT SHAPE:", gradients.shape)

        activations, gradients = self._squeeze_batch_dim(
            activations, gradients
        )

        if activations.shape != gradients.shape:
            raise ValueError(
                "Activation and gradient shapes must match after "
                f"batch handling: {activations.shape} vs {gradients.shape}"
            )

        print("ACTIVATION SHAPE (after batch):", activations.shape)
        print("GRADIENT SHAPE (after batch):", gradients.shape)

        if activations.ndim == 4:
            activations, gradients = self._squeeze_batch_dim(
                activations, gradients
            )
            if activations.ndim == 4:
                raise ValueError(
                    f"Unsupported 4D activation shape: {activations.shape}"
                )

        if activations.ndim == 3:
            if self._is_batch_token_channel_map(activations):
                activations, gradients = activations[0], gradients[0]
                return self._tokens_to_spatial(activations, gradients)

            if self._is_channels_first_spatial(activations):
                return activations, gradients

            if self._is_hwc_spatial(activations):
                activations = np.transpose(activations, (2, 0, 1))
                gradients = np.transpose(gradients, (2, 0, 1))
                return activations, gradients

            raise ValueError(
                f"Unrecognized 3D activation layout: {activations.shape}"
            )

        if activations.ndim == 2:
            if self._is_token_channel_map(activations):
                return self._tokens_to_spatial(activations, gradients)

            activations = activations[:, np.newaxis, np.newaxis]
            gradients = gradients[:, np.newaxis, np.newaxis]
            return activations, gradients

        raise ValueError(
            f"Unexpected activation shape for GradCAM: {activations.shape}"
        )

    def generate_cam(self, input_tensor, target_class=None):
        self.gradients = None
        self.activations = None

        was_training = self.model.training
        grad_flags = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        input_tensor = input_tensor.to(
            next(self.model.parameters()).device
        )
        input_tensor.requires_grad_(True)

        try:
            with torch.enable_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).squeeze()

                if target_class is None:
                    target_class = 1 if probability.item() > 0.5 else 0

                self.model.zero_grad(set_to_none=True)

                loss = probability if target_class == 1 else (1 - probability)
                loss.backward(retain_graph=False)

                grad_tensor = self.gradients
                if grad_tensor is None and isinstance(self.activations, torch.Tensor):
                    grad_tensor = self.activations.grad

                if grad_tensor is None:
                    print("WARNING: Gradients not found.")
                    return None

                if self.activations is None:
                    raise ValueError(
                        "Activations were not captured. "
                        "Check target layer and hooks."
                    )

                gradients = grad_tensor.detach().cpu().numpy()
                activations = self.activations.detach().cpu().numpy()

                activations, gradients = self._to_spatial_maps(
                    activations, gradients
                )

                weights = np.mean(
                    np.maximum(gradients, 0),
                    axis=(1, 2)
                )

                cam = np.zeros(activations.shape[1:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * activations[i]

                cam = np.maximum(cam, 0)
                cam = cam - np.min(cam)
                cam = cam / (np.max(cam) + 1e-8)
                cam = np.power(cam, 2.0)

                cam = cv2.resize(
                    cam,
                    (GRADCAM_OVERLAY_SIZE, GRADCAM_OVERLAY_SIZE),
                    interpolation=cv2.INTER_CUBIC,
                )
                
                return enhance_cam(cam)
        finally:
            for param, requires_grad in zip(
                self.model.parameters(), grad_flags
            ):
                param.requires_grad = requires_grad
            if was_training:
                self.model.train()


# =====================================
# CAM ENHANCEMENT & VISUAL BUILDERS
# =====================================

def enhance_cam(cam):

    # =====================================
    # POSITIVE ACTIVATIONS ONLY
    # =====================================

    cam = np.maximum(cam, 0)

    # =====================================
    # RADIAL SUPPRESSION
    # Reduce corner bias
    # =====================================

    h, w = cam.shape

    yy, xx = np.mgrid[0:h, 0:w]

    center_y = h / 2
    center_x = w / 2

    dist = np.sqrt(
        ((yy - center_y) / center_y) ** 2 +
        ((xx - center_x) / center_x) ** 2
    )

    radial_mask = 1 - 0.25 * dist

    radial_mask = np.clip(radial_mask, 0.7, 1.0)

    cam = cam * radial_mask

    # =====================================
    # BORDER SUPPRESSION
    # Remove edge artifacts
    # =====================================

    border = 6

    cam[:border, :] *= 0.3
    cam[-border:, :] *= 0.3

    cam[:, :border] *= 0.3
    cam[:, -border:] *= 0.3

    # =====================================
    # NORMALIZE
    # =====================================

    cam = cam - np.min(cam)

    cam = cam / (np.max(cam) + 1e-8)

    # =====================================
    # EDGE-PRESERVING SMOOTHING
    # Better than GaussianBlur for pathology
    # =====================================

    cam = cv2.bilateralFilter(
        cam.astype(np.float32),
        7,
        0.1,
        5
    )

    # =====================================
    # ADAPTIVE THRESHOLD
    # Dynamic focus control
    # =====================================

    activation_strength = np.mean(cam)

    if activation_strength > 0.35:
        threshold_percent = 88

    elif activation_strength > 0.20:
        threshold_percent = 82

    else:
        threshold_percent = 72

    threshold = np.percentile(
        cam,
        threshold_percent
    )

    cam = np.where(
        cam >= threshold,
        cam,
        0
    )

    # =====================================
    # SMALL ARTIFACT REMOVAL
    # Remove tiny noisy hotspots
    # =====================================

    binary_map = (cam > 0).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_map,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(cam)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 40:
            cleaned[labels == i] = cam[labels == i]

    cam = cleaned

    cam = np.power(cam, 0.8)

    # =====================================
    # FINAL NORMALIZATION
    # =====================================

    cam = cam - np.min(cam)

    cam = cam / (np.max(cam) + 1e-8)

    return cam.astype(np.float32)

def cam_to_heatmap(cam, size=None):
    size = size or GRADCAM_OVERLAY_SIZE
    cam_hi = cv2.resize(
        cam,
        (size, size),
        cv2.INTER_CUBIC
    )
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_hi),
        cv2.COLORMAP_TURBO,
    )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (3, 3),
        0
    )
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def build_overlay(original, heatmap, cam):

    # CAM -> 3 channels
    cam_3d = np.stack(
        [cam] * 3,
        axis=-1
    )

    # Dynamic alpha map
    alpha = 0.15 + 0.55 * cam_3d

    overlay = (
        original.astype(np.float32) * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    )

    return overlay.astype(np.uint8)


def _save_rgb_image(path, image):
    Image.fromarray(image.astype(np.uint8)).save(path)


def _save_grayscale_map(path, cam, cmap_name="turbo"):
    plt.imsave(path, cam, cmap=cmap_name, vmin=0, vmax=1)


def _stem_from_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


# =====================================
# MODEL LOADING
# =====================================

def load_gradcam_model(model_type="fusion"):
    if model_type == "fusion":
        model = FeatureFusionModel()
    elif model_type == "vit":
        model = ViTModel()
    elif model_type == "swin":
        model = SwinModel()
    elif model_type == "convnext":
        model = ConvNeXtModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = "checkpoints/best_model.pth"
    if os.path.isfile(checkpoint):
        model.load_state_dict(
            torch.load(checkpoint, map_location=CFG.DEVICE)
        )

    model = model.to(CFG.DEVICE)
    model.eval()
    return model


# =====================================
# VISUALIZE
# =====================================

def visualize_gradcam(
    image_path,
    model=None,
    backbone="fusion",
    model_type="fusion",
    gradcam=None,
    show_plot=False,
    output_dir=GRADCAM_GALLERY_DIR,
):
    """
    Generate publication-quality GradCAM visuals for one image.

    Saves a 6-panel composite plus individual heatmap, overlay, and
    enhanced outputs under output_dir (default: figures/gradcam_gallery/).
  """
    if model is None:
        model = load_gradcam_model(model_type=model_type)

    if gradcam is None:
        target_layer = get_target_layer(model, backbone=backbone)
        gradcam = GradCAM(model, target_layer)

    try:
        original, input_tensor = load_image(image_path)
        input_tensor = input_tensor.to(CFG.DEVICE)

        cam = gradcam.generate_cam(input_tensor)
        if cam is None:
            print(f"SKIPPED (no gradients): {image_path}")
            return None

        heatmap = cam_to_heatmap(cam)
        overlay = build_overlay(
            original,
            heatmap,
            cam
        )

        edges = cv2.Canny(
            cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
            50,
            150
        )

        overlay[edges > 0] = (
            0.85 * overlay[edges > 0] +
            0.15 * np.array([255,255,255])
        )
        

        # Inference only — keep no_grad here; GradCAM uses generate_cam()
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).squeeze()
            prob_value = float(probability.item())
            pred = 1 if prob_value > 0.5 else 0

        class_name = "IDC Positive" if pred == 1 else "IDC Negative"
        confidence = prob_value if pred == 1 else (1.0 - prob_value)

        os.makedirs(output_dir, exist_ok=True)
        stem = _stem_from_path(image_path)

       
        

        panel_title = (
            f"{class_name}  ({confidence * 100:.1f}%)"
        )

        panels = [
            (original, None, "Original"),
            (overlay, None, "Model Attention Map"),
        ]

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(7, 3.5),
        )
        if len(panels) == 1:
            axes = [axes]

        for ax, (image, cmap, title) in zip(axes, panels):
            if cmap is not None:
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(image)
            ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
            ax.axis("off")

        fig.suptitle(
            panel_title,
            fontsize=24,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{stem}_panel.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        if show_plot:
            plt.show()
        plt.close(fig)

        

        return True

    except Exception as exc:
        print(f"ERROR processing {image_path}: {exc}")
        plt.close("all")
        return None


# =====================================
# GRADCAM GALLERY
# =====================================

def generate_gradcam_gallery(
    backbone="fusion",
    model_type="fusion",
    num_samples=GRADCAM_GALLERY_SAMPLES,
    output_dir=GRADCAM_GALLERY_DIR,
):
    positive_images = glob.glob(
        "project_data/final_dataset/test/IDC_positive/*.png"
    )
    negative_images = glob.glob(
        "project_data/final_dataset/test/IDC_negative/*.png"
    )
    random.seed(42)
    random.shuffle(positive_images)
    random.shuffle(negative_images)
    positive_images = positive_images[:4]
    negative_images = negative_images[:4]

    per_class = max(1, num_samples // 2)
    selected_images = (
        positive_images
        + negative_images
    )

    print(
        f"\nGENERATING GRADCAM GALLERY "
        f"({len(selected_images)} samples) -> {output_dir}\n"
    )

    os.makedirs(output_dir, exist_ok=True)

    model = load_gradcam_model(model_type=model_type)
    target_layer = get_target_layer(model, backbone=backbone)
    gradcam = GradCAM(model, target_layer)

    saved = 0
    failed = 0

    for image_path in selected_images:
        print(f"PROCESSING:\n{image_path}")
        try:
            result = visualize_gradcam(
                image_path,
                model=model,
                backbone=backbone,
                model_type=model_type,
                gradcam=gradcam,
                output_dir=output_dir,
            )
            if result is None:
                failed += 1
            else:
                saved += 1
        except Exception as exc:
            failed += 1
            print(f"GALLERY ERROR (continuing): {image_path}\n  {exc}")

    gradcam.remove_hooks()

    print(
        f"\nGRADCAM GALLERY COMPLETE — "
        f"saved: {saved}, skipped/failed: {failed}"
    )

    # =====================================
    # FINAL GRID
    # =====================================

    all_panels = sorted(
        glob.glob(os.path.join(output_dir, "*_panel.png"))
    )

    fig, axes = plt.subplots(4, 2, figsize=(10, 18))

    axes = axes.flatten()

    for ax, img_path in zip(axes, all_panels):

        img = Image.open(img_path)

        ax.imshow(img)

        ax.axis("off")

    for ax in axes[len(all_panels):]:
        ax.axis("off")

    plt.tight_layout()

    final_grid_path = os.path.join(
        output_dir,
        "FINAL_GRADCAM_GRID.png"
    )

    plt.savefig(
        final_grid_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nFINAL GRID SAVED:\n{final_grid_path}")
# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    generate_gradcam_gallery(

        backbone="convnext",

        model_type="fusion",

        num_samples=8
    )
    print("\nFINAL CLEAN GRADCAM GALLERY READY\n")
    print("IMPROVED GRADCAM READY")

