# -*- coding: utf-8 -*-

"""
Publication-quality GradCAM for hybrid breast cancer histopathology classification.

Uses the full FeatureFusionModel for inference, but attributes spatial attention
only through ConvNeXt backbone features (stages[-1]). ViT/Swin branches are ignored
for visualization.

Visualization modes (toggle STRONG_PUBLICATION_MODE at top of file):
  - NORMAL_MODE: soft TURBO overlays, tissue-preserving alpha
  - STRONG_PUBLICATION_MODE: JET hotspots, contrast boost, thesis-style figures

Run:
    python evaluation/gradcam_main.py
"""

from __future__ import annotations

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Project root on path when executed as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import CFG
from models.feature_fusion import FeatureFusionModel


# ---------------------------------------------------------------------------
# Paths and manually selected images (no dataset scanning)
# ---------------------------------------------------------------------------

IMAGE_SIZE = CFG.IMAGE_SIZE
DEVICE = CFG.DEVICE
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures", "gradcam_final")

SELECTED_IMAGES = [
    os.path.join(PROJECT_ROOT, "evaluation", "selected_images", "positive_1.png"),
    os.path.join(PROJECT_ROOT, "evaluation", "selected_images", "positive_2.png"),
    os.path.join(PROJECT_ROOT, "evaluation", "selected_images", "negative_1.png"),
    os.path.join(PROJECT_ROOT, "evaluation", "selected_images", "negative_2.png"),
]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------------
# Visualization mode (switch here)
# ---------------------------------------------------------------------------
# NORMAL_MODE: soft, tissue-preserving overlays for scientific interpretation
# STRONG_PUBLICATION_MODE: bold JET hotspots for thesis / jury figures
STRONG_PUBLICATION_MODE = True
NORMAL_MODE = not STRONG_PUBLICATION_MODE

BLOB_GAUSSIAN_KERNEL = (51, 51)
BLOB_PERCENTILE = 90
BLOB_POWER = 2.5
BLOB_MIN_THRESHOLD = 0.25

# Optional light smoothing in normal mode only (set to 0 to disable)
GAUSSIAN_KERNEL = 3


# ---------------------------------------------------------------------------
# Image and model helpers
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> tuple[np.ndarray, torch.Tensor]:
    """Load RGB display image and normalized model input tensor."""
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original = np.array(image_resized)
    tensor = TRANSFORM(image_resized).unsqueeze(0)
    return original, tensor


def load_model() -> FeatureFusionModel:
    """Load hybrid fusion model from the best checkpoint."""
    model = FeatureFusionModel()
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_convnext_target_layer(model: FeatureFusionModel) -> torch.nn.Module:
    """Final ConvNeXt stage - spatial feature maps for GradCAM."""
    return model.convnext_model.backbone.stages[-2]

# ---------------------------------------------------------------------------
# GradCAM (ConvNeXt spatial maps only)
# ---------------------------------------------------------------------------

class GradCAM:
    """Minimal GradCAM with forward/backward hooks on a spatial ConvNeXt layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def remove_hooks(self) -> None:
        """Detach hooks to avoid memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def _save_activation(self, module, inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output
        if isinstance(self.activations, torch.Tensor) and self.activations.requires_grad:
            self.activations.retain_grad()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        if grad_output is not None and grad_output[0] is not None:
            self.gradients = grad_output[0]

    @staticmethod
    def _to_channels_first(
        activations: np.ndarray,
        gradients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure (C, H, W) layout for ConvNeXt spatial tensors."""
        if activations.ndim == 4:
            activations = activations[0]
            gradients = gradients[0]

        if activations.ndim != 3:
            raise ValueError(f"Expected 3D spatial map, got {activations.shape}")

        # (H, W, C) -> (C, H, W); ConvNeXt may already be (C, H, W)
        if activations.shape[0] <= 32:
            activations = np.transpose(activations, (2, 0, 1))
            gradients = np.transpose(gradients, (2, 0, 1))

        return activations, gradients

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray | None:
        """Compute a normalized CAM in [0, 1] resized to IMAGE_SIZE."""
        self.gradients = None
        self.activations = None

        was_training = self.model.training
        grad_flags = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)

        try:
            with torch.enable_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).squeeze()

                if target_class is None:
                    target_class = 1 if probability.item() > 0.5 else 0

                self.model.zero_grad(set_to_none=True)
                loss = probability if target_class == 1 else (1.0 - probability)
                loss.backward(retain_graph=False)

                grad_tensor = self.gradients
                if grad_tensor is None and isinstance(self.activations, torch.Tensor):
                    grad_tensor = self.activations.grad
                if grad_tensor is None or self.activations is None:
                    return None

                activations = self.activations.detach().cpu().numpy()
                gradients = grad_tensor.detach().cpu().numpy()
                activations, gradients = self._to_channels_first(activations, gradients)

                # Global average pooling of gradients -> channel weights
                weights = np.mean(gradients, axis=(1, 2))

                cam = np.zeros(activations.shape[1:], dtype=np.float32)
                for channel, weight in enumerate(weights):
                    cam += weight * activations[channel]

                cam = np.maximum(cam, 0.0)
                cam -= cam.min()
                cam /= cam.max() + 1e-8

                cam = cv2.resize(
                    cam,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_CUBIC,
                )
                cam = np.maximum(cam, 0.0)
                cam -= cam.min()
                cam /= cam.max() + 1e-8
                # Display enhancements applied in postprocess_cam()
                return cam.astype(np.float32)
        finally:
            for param, requires_grad in zip(self.model.parameters(), grad_flags):
                param.requires_grad = requires_grad
            if was_training:
                self.model.train()


# ---------------------------------------------------------------------------
# Visualization (mode-specific display only; GradCAM math unchanged)
# ---------------------------------------------------------------------------

def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Map CAM to [0, 1] after ReLU."""
    cam = np.maximum(cam, 0.0).astype(np.float32)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam


def postprocess_cam(cam: np.ndarray) -> np.ndarray:
    """
    Prepare raw GradCAM for display.

    NORMAL_MODE: light Gaussian smoothing, no thresholding or power transforms.
    STRONG_PUBLICATION_MODE: contrast boost, soft threshold, light sharpen blur.
    """
    cam = _normalize_cam(cam)

    if STRONG_PUBLICATION_MODE:

        threshold = np.percentile(cam, BLOB_PERCENTILE)

        mask = cam >= threshold

        blob_map = cv2.GaussianBlur(
            mask.astype(np.float32),
            BLOB_GAUSSIAN_KERNEL,
            0
        )

        blob_map = np.power(blob_map, BLOB_POWER)

        blob_map = _normalize_cam(blob_map)

        blob_map[blob_map < BLOB_MIN_THRESHOLD] = 0

        return blob_map 

    # Normal mode: preserve soft, localized, morphology-friendly maps
    if GAUSSIAN_KERNEL > 0:
        kernel = GAUSSIAN_KERNEL | 1
        cam = cv2.GaussianBlur(cam, (kernel, kernel), 0)
        cam = _normalize_cam(cam)
    return cam


def get_colormap() -> int:
    """Colormap used for heatmap export and overlays."""
    if STRONG_PUBLICATION_MODE:
        return cv2.COLORMAP_JET
    return cv2.COLORMAP_TURBO


def get_overlay_alpha() -> float:
    """Heatmap blend strength (higher = more visually dominant overlay)."""
    if STRONG_PUBLICATION_MODE:
        return 0.70
    return 0.28


def cam_to_heatmap(cam: np.ndarray) -> np.ndarray:
    """Apply mode-specific colormap to the display-ready CAM."""
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), get_colormap())
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def build_overlay(original: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend heatmap onto histopathology (natural or publication-strong)."""
    heatmap = cam_to_heatmap(cam)
    cam_3d = np.stack([cam, cam, cam], axis=-1)
    base = original.astype(np.float32)

    if STRONG_PUBLICATION_MODE:
        # Dim non-activated tissue so hotspots stand out
        background_factor = 1.0 - (0.35 * cam_3d)
        base = base * background_factor

    alpha = get_overlay_alpha() * cam_3d
    blended = base * (1.0 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def predict_label_and_confidence(
    model: FeatureFusionModel,
    input_tensor: torch.Tensor,
) -> tuple[str, float, int]:
    """Run hybrid model inference (all branches)."""
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        prob = float(torch.sigmoid(output).squeeze().item())

    pred = 1 if prob > 0.5 else 0
    confidence = prob if pred == 1 else (1.0 - prob)
    label = "IDC Positive" if pred == 1 else "IDC Negative"
    return label, confidence, pred


def save_panel_figure(
    original: np.ndarray,
    overlay: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Save a clean 1x2 panel: original | GradCAM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")

    for ax, image, subtitle in zip(
        axes,
        [original, overlay],
        ["Original", "GradCAM Overlay"],
    ):
        ax.imshow(image)
        ax.set_title(subtitle, fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_image(
    image_path: str,
    model: FeatureFusionModel,
    gradcam: GradCAM,
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Generate and save GradCAM outputs for one selected image."""
    if not os.path.isfile(image_path):
        print(f"SKIP (missing): {image_path}")
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    original, input_tensor = load_image(image_path)
    input_tensor = input_tensor.to(DEVICE)

    label, confidence, pred = predict_label_and_confidence(model, input_tensor)
    cam_raw = gradcam.generate_cam(input_tensor, target_class=pred)
    if cam_raw is None:
        print(f"SKIP (no gradients): {image_path}")
        return None

    cam = postprocess_cam(cam_raw)
    heatmap = cam_to_heatmap(cam)
    overlay = build_overlay(original, cam)
    title = f"{label} ({confidence * 100:.1f}%)"

    original_path = os.path.join(output_dir, f"{stem}_original.png")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    heatmap_path = os.path.join(output_dir, f"{stem}_heatmap.png")
    cam_path = os.path.join(output_dir, f"{stem}_cam.png")
    panel_path = os.path.join(output_dir, f"{stem}_panel.png")

    Image.fromarray(original).save(original_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray((cam * 255).astype(np.uint8), mode="L").save(cam_path)
    save_panel_figure(original, overlay, title, panel_path)

    print(f"Saved: {stem} -> {title}")
    return original, overlay, title


def build_final_grid(
    rows: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: str,
) -> None:
    """Assemble a 4x2 publication grid with prediction labels above each row."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 18), facecolor="white")

    if len(rows) == 1:
        axes = np.array([axes])

    for row_idx, (original, overlay, title) in enumerate(rows):
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].set_title("Original", fontsize=12, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title("GradCAM Overlay", fontsize=12, fontweight="bold")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.30)

    for row_idx, (_, _, title) in enumerate(rows):
        pos_left = axes[row_idx, 0].get_position()
        pos_right = axes[row_idx, 1].get_position()
        center_x = (pos_left.x0 + pos_right.x1) / 2
        top_y = pos_left.y1 + 0.012
        fig.text(
            center_x,
            top_y,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not SELECTED_IMAGES:
        raise ValueError("No selected images found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode_label = "STRONG_PUBLICATION" if STRONG_PUBLICATION_MODE else "NORMAL"
    print(f"Visualization mode: {mode_label}")

    print("Loading hybrid model...")
    model = load_model()

    target_layer = get_convnext_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    grid_rows: list[tuple[np.ndarray, np.ndarray, str]] = []

    try:
        print(f"Processing {len(SELECTED_IMAGES)} selected images...")
        for image_path in SELECTED_IMAGES:
            result = process_image(image_path, model, gradcam, OUTPUT_DIR)
            if result is not None:
                grid_rows.append(result)

        if grid_rows:
            grid_path = os.path.join(OUTPUT_DIR, "FINAL_GRADCAM_GRID.png")
            build_final_grid(grid_rows, grid_path)
            print(f"\nFinal grid saved:\n  {grid_path}")
        else:
            print("\nNo images processed. Check evaluation/selected_images/.")
    finally:
        gradcam.remove_hooks()

    print("\nGradCAM pipeline complete.")


if __name__ == "__main__":
    main()
