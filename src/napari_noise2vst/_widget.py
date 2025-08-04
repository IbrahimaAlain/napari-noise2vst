#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Noise2VST Napari Widget: A denoising plugin for napari based on Noise2VST.

This plugin allows training and inference of the Noise2VST model within the napari viewer.
"""

import os
import sys
import torch
import numpy as np
import traceback
import csv
import re
import textwrap
import matplotlib.pyplot as plt
import napari
from pathlib import Path
from skimage.util import img_as_float
from typing import TYPE_CHECKING
from magicgui.widgets import Container, create_widget, PushButton, Label, Slider, ProgressBar
from qtpy.QtWidgets import QFileDialog

if TYPE_CHECKING:
    import napari

# Plugin directories setup
PLUGIN_DIR = Path(__file__).parent
WEIGHTS_DIR = PLUGIN_DIR / "pretrained_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# Add local noise2vst repo to sys.path to enable imports
repo_path = PLUGIN_DIR / "noise2vst"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

# Import models and utilities from noise2vst
from noise2vst.models.noise2vst import Noise2VST, Spline
from noise2vst.models.ffdnet import FFDNet
from noise2vst.models.drunet import DRUNet
from noise2vst.utilities.utilities import f_GAT, f_GAT_inv

# Attempt import for pretrained weights downloader; fallback to None if not available
try:
    from napari_noise2vst.pretrained_weights.download import download_weights
except ImportError:
    download_weights = None


class Noise2VSTWidget(Container):
    """
    Napari plugin widget for Noise2VST denoising.

    Provides a GUI to load an image, train the model, run inference, visualize and export splines.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Noise2VST model and move to device
        self.model = Noise2VST().to(self.device)

        # --- GUI Widgets ---

        # Input image selector
        self.input_label = Label(value="Input Image:")
        self.image_input = create_widget(label="Input Image", annotation="napari.layers.Image")
        self.viewer.layers.selection.events.changed.connect(self.sync_input_image_with_selection)


        # Step 1: Training controls
        self.step1_label = Label(value="STEP 1: TRAIN")
        self.iter_slider = Slider(
            label="Number of training iterations:",
            value=2000,
            min=100,
            max=5000,
            step=100,
        )
        self.train_button = PushButton(label="Fit the VST model")
        self.progress_bar = ProgressBar(min=0, max=100, label="Progress", visible=False)

        # Container for training step
        self.step1_container = Container(widgets=[
            self.step1_label,
            self.iter_slider,
            self.train_button,
            self.progress_bar,
        ])

        # Step 2: Prediction and evaluation controls
        self.step2_label = Label(value="STEP 2: PREDICT & EVALUATE")
        self.eval_button = PushButton(label="Run Denoising")
        self.run_denoise_progress = ProgressBar(min=0, max=100, visible=False)
        self.plot_spline_button = PushButton(label="Visualize VST")
        self.save_spline_button = PushButton(label="Save Spline Knots")

        # Container for evaluation step (hidden until training completes)
        self.step2_container = Container(widgets=[
            self.step2_label,
            self.eval_button,
            self.run_denoise_progress,
            self.plot_spline_button,
            self.save_spline_button,
        ])
        self.step2_container.visible = False

        # Status display label
        self.status = Label(value="Status: Ready")

        # Connect callbacks
        self.train_button.changed.connect(self.train_model)
        self.eval_button.changed.connect(self.evaluate_model)
        self.plot_spline_button.changed.connect(self.plot_spline)
        self.save_spline_button.changed.connect(self.export_spline_knots)

        # Add all widgets to this container
        self.extend([
            self.input_label,
            self.image_input,
            self.step1_container,
            self.step2_container,
            self.status,
        ])

        # Attempt to download pretrained weights on initialization if possible
        if download_weights is not None:
            self.update_status("Checking pretrained weights...")
            try:
                download_weights()
                self.update_status("Pretrained weights ready.")
            except Exception as e:
                self.update_status(f"Automatic weight download failed: {e}")
        else:
            self.update_status("No automatic download function available.")

    def _info(self, msg: str):
        print(f"[INFO] {msg}")
        self.status.value = self.wrap_status_text(f"Status: {msg}")

    def _error(self, msg: str):
        print(f"[ERROR] {msg}")
        self.status.value = self.wrap_status_text(f"Status: {msg}")


    def update_status(self, message: str):
        """
        Update the status label with wrapped text for readability.
        """
        self.status.value = self.wrap_status_text(f"Status: {message}")
        print(f"[STATUS] {message}")

    def wrap_status_text(self, text: str, max_length: int = 40) -> str:
        """
        Wrap the status text to a maximum length per line without removing characters.

        Args:
            text: The string to wrap.
            max_length: Maximum characters per line.

        Returns:
            Wrapped text with newlines inserted.
        """
        return "\n".join(textwrap.wrap(text, width=max_length))
    
    def sync_input_image_with_selection(self):
        """
        Automatically update the input image field when a new image layer is selected in the viewer.
        """
        if not self.viewer.layers:
            return

        selected = self.viewer.layers.selection
        if selected:
            layer = next(iter(selected))
            if isinstance(layer, napari.layers.Image):
                self.image_input.value = layer
                self.update_status(f"Image input set to: {layer.name}")    

    def _get_image_data(self):
        """
        Retrieve the numpy image data from the selected napari image layer.

        Returns:
            Image data as float numpy array or None if no image selected.
        """
        img_layer = self.image_input.value
        if img_layer is None:
            self.update_status("No image selected.")
            return None
        return img_as_float(img_layer.data)

    def load_ffdnet_model(self, image: torch.Tensor):
        """
        Load FFDNet model based on image color channels.

        Args:
            image: Input image tensor (batch x channels x height x width).

        Returns:
            FFDNet model in eval mode on the selected device.
        """
        is_color = image.shape[1] == 3
        ffdnet_path = WEIGHTS_DIR / ("ffdnet_color.pth" if is_color else "ffdnet_gray.pth")
        ffdnet = FFDNet(color=is_color).to(self.device).requires_grad_(False)
        ffdnet.load_state_dict(torch.load(ffdnet_path, map_location=self.device), strict=True)
        return ffdnet


    def load_drunet_model(self, image: torch.Tensor):
        """
        Load DRUNet model based on image color channels.

        Args:
            image: Input image tensor (batch x channels x height x width).

        Returns:
            DRUNet model in eval mode on the selected device.
        """
        is_color = image.shape[1] == 3
        drunet_path = WEIGHTS_DIR / ("drunet_color.pth" if is_color else "drunet_gray.pth")
        drunet = DRUNet(color=is_color).to(self.device).requires_grad_(False)
        drunet.load_state_dict(torch.load(drunet_path, map_location=self.device), strict=True)
        return drunet

    def train_model(self, _=None):
        """
        Train the Noise2VST model using the input image and selected number of iterations.
        Updates progress bar and status messages.
        """
        image = self._get_image_data()
        if image is None:
            return

        # Preprocess input image into tensor with shape (batch, channels, height, width)
        if image.ndim == 2:
            image = image[None, None, :, :]
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)[None, :]
        elif image.ndim == 4:
            pass  # Already in batch format
        else:
            self.update_status(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        # Ensure pretrained weights are downloaded before training
        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self.update_status(f"Download failed: {e}")
                return

        # Load FFDNet model for training
        try:
            ffdnet = self.load_ffdnet_model(image)
        except Exception as e:
            self.update_status(f"Model loading failed: {e}")
            return

        # Compose spline weights path based on selected image name
        image_layer = self.image_input.value
        if image_layer is None:
            self.update_status("No input image selected. Cannot load/save weights.")
            return

        image_name = image_layer.name
        spline_path = WEIGHTS_DIR / f"noise2vst_spline_{image_name}.pth"

        # Load spline weights if exist for this image
        if spline_path.exists():
            try:
                self.model.load_state_dict(torch.load(spline_path, map_location=self.device))
                self.update_status(f"Spline weights loaded for image '{image_name}'.")
            except Exception as e:
                self.update_status(f"Failed to load spline weights: {e}")

        try:
            self.progress_bar.visible = True
            self.progress_bar.value = 0
            self.update_status("Training started...")
            nb_iter = self.iter_slider.value

            # Train model with progress callback updating progress bar
            self.model.fit(
                image,
                ffdnet,
                nb_iterations=nb_iter,
                progress_callback=lambda v: setattr(self.progress_bar, "value", v)
            )

            self.progress_bar.visible = False
            self.step2_container.visible = True
            self.update_status("Training complete.")
        except Exception as e:
            self.progress_bar.visible = False
            self.update_status(f"Training failed: {e}")
            traceback.print_exc()
            return

        # Save trained spline weights with image-specific filename
        try:
            torch.save(self.model.state_dict(), spline_path)
            self.update_status(f"Weights saved to {spline_path}")
        except Exception as e:
            self.update_status(f"Failed to save weights: {e}")

    def evaluate_model(self, _=None):
        """
        Run denoising inference on the selected input image.
        Adds or updates the denoised image layer in napari viewer.
        """
        image = self._get_image_data()
        if image is None:
            return

        # Preprocess input image into tensor shape (batch, channels, height, width)
        if image.ndim == 2:
            image = image[None, None, :, :]
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)[None, :]
        elif image.ndim == 4:
            pass
        else:
            self.update_status(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        # Download weights if needed
        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self.update_status(f"Download failed: {e}")
                return

        # Load DRUNet model for inference
        try:
            drunet = self.load_drunet_model(image)
        except Exception as e:
            self.update_status(f"Model loading failed: {e}")
            return

        # Load spline weights specific to the selected image (if any)
        image_layer = self.image_input.value
        if image_layer is None:
            self.update_status("No input image selected. Cannot load spline weights.")
            return

        image_name = image_layer.name
        spline_path = WEIGHTS_DIR / f"noise2vst_spline_{image_name}.pth"

        if spline_path.exists():
            try:
                self.model.load_state_dict(torch.load(spline_path, map_location=self.device))
                self.update_status(f"Spline weights loaded for image '{image_name}'.")
            except Exception as e:
                self.update_status(f"Failed to load spline weights: {e}")
                return
        else:
            self.update_status(f"No spline weights found for image '{image_name}', running inference with default model.")

        try:
            self.run_denoise_progress.visible = True
            self.run_denoise_progress.value = 0
            with torch.no_grad():
                output = self.model(image, drunet)

                self.run_denoise_progress.value = 100
                self.run_denoise_progress.visible = False

                # Squeeze batch dimension if present
                if output.dim() == 4 and output.shape[0] == 1:
                    output = output.squeeze(0)

                output = output.permute(1, 2, 0).cpu().numpy()

                # Determine if output is grayscale or RGB for napari display
                if output.shape[2] == 1:
                    output = output[..., 0]
                    rgb_flag = False
                else:
                    rgb_flag = True
        except Exception as e:
            self.run_denoise_progress.visible = False
            self.update_status(f"Inference failed: {e}")
            traceback.print_exc()
            return

        # Compose new layer name based on input image name
        denoised_name = f"{image_name}_denoised"
        colormap = getattr(image_layer.colormap, "name", image_layer.colormap)
        contrast_limits = (float(output.min()), float(output.max()))
        gamma = image_layer.gamma or 1.0

        if denoised_name in self.viewer.layers:
            self.viewer.layers[denoised_name].data = output

        else:

            self.viewer.add_image(output, name=denoised_name,
                                  rgb=rgb_flag, colormap=colormap,
                                  contrast_limits=contrast_limits,
                                  gamma=gamma)

        self.update_status("Denoising complete.")


    def plot_spline(self, _=None):
        """
        Plot the learned VST splines (forward and inverse) for the currently selected input image.
        The figure is customized and saved as a PNG named after the image.
        """
        try:
            image_layer = self.image_input.value
            if image_layer is None:
                self.update_status("No image selected for spline plotting.")
                return

            base_name = image_layer.name
            if base_name.endswith("_denoised"):
                base_name = base_name.removesuffix("_denoised")

            safe_name = re.sub(r"[^\w.-]", "_", base_name)
            spline_path = WEIGHTS_DIR / f"noise2vst_spline_{safe_name}.pth"
            if not spline_path.exists():
                self.update_status(f"Spline weights file does not exist for image '{image_layer.name}'.")
                return

            # Load spline weights for the selected image
            self.model.load_state_dict(torch.load(spline_path, map_location=self.device))

            with torch.no_grad():
                x = torch.linspace(0, 1, self.model.spline1.nb_knots, device=self.device)
                y = self.model.spline1(x)

                if getattr(self.model, "inverse", False):
                    z = self.model.spline1(y, inverse=True)
                else:
                    z = self.model.spline2(x)

                # Convert to CPU for plotting
                x_cpu, y_cpu, z_cpu = x.cpu(), y.cpu(), z.cpu()
                c = y_cpu.min()

                # Prepare custom figure with personalized name
                fig_title = f"VST Splines - {image_layer.name}"
                fig = plt.figure(num=fig_title, figsize=(8, 4))
                fig.clf()  # Clear figure if exists

                if getattr(self.model, "inverse", False):
                    plt.plot(x_cpu, y_cpu - c, color='blue', label=r"$f_\theta$")
                    plt.plot(y_cpu - c, z_cpu, color='orange', label=r"$f^{inv}_{\theta, \alpha, \beta}$")
                else:
                    plt.plot(x_cpu, y_cpu - c, color='blue', label=r"$f_{\theta_1}$")
                    plt.plot(x_cpu - c, z_cpu, color='orange', label=r"$f_{\theta_2}$")

                plt.plot(x_cpu, x_cpu, "--", color="black", label="identity")
                plt.xlabel("Input")
                plt.ylabel("Output")
                plt.title(fig_title)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # Save the plot in outputs folder
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / f"spline_plot_{safe_name}.png"
                plt.savefig(save_path, dpi=150)
                plt.show()

        except Exception as e:
            self.update_status(f"Failed to plot spline: {e}")
            import traceback
            traceback.print_exc()

    
    def theta2y(self, theta):
        if not Spline().is_strictly_increasing:
            return theta
        theta0, theta1 = torch.split(theta, [1, Spline().nb_knots-1], dim=0)
        return torch.cumsum(torch.cat((theta0, theta1.exp() + Spline().eps), dim=0), dim=0)



    def export_spline_knots(self, _=None):
        """Export spline values (x, y - c, y - c, z) as CSV for the selected image.

        If the model is in inverse mode, z corresponds to f⁻¹(y - c), otherwise z = f₂(x).
        """

        image_layer = self.image_input.value
        if not image_layer:
            self._error("No image selected.")
            return
        
        base_name = image_layer.name
        if base_name.endswith("_denoised"):
            base_name = base_name.removesuffix("_denoised")
        safe_name = re.sub(r"[^\w.-]", "_", base_name)
        spline_path = WEIGHTS_DIR / f"noise2vst_spline_{safe_name}.pth"

        if not spline_path.exists():
            self._error(f"No spline weights found for '{image_layer.name}'. Train the model first.")
            return

        try:
            weights = torch.load(spline_path, map_location=self.device)
            self.model.load_state_dict(weights)

            theta_in = weights.get("spline1.theta")
            theta_out = weights.get("spline2.theta")

            if theta_in is None:
                self._error("spline1.theta not found in weights.")
                return

            y_in = self.theta2y(theta_in)
            x = torch.linspace(0, 1, len(y_in), device=self.device)
            c = y_in.min()
            y_centered = y_in - c

            with torch.no_grad():
                if self.model.inverse:
                    z = self.model.spline1(y_in, inverse=True)
                else:
                    z = self.model.spline2(x)

            x = x.cpu().numpy()
            y_centered = y_centered.cpu().numpy()
            z = z.cpu().numpy()

            header = ["x", "y", "x_inv", "y_inv"]
            rows = zip(x, y_centered, y_centered, z)


            path, _ = QFileDialog.getSaveFileName(
                caption="Export Spline Knots",
                filter="CSV Files (*.csv)",
                directory=f"spline_knots_{safe_name}.csv"
            )
            if not path:
                self._info("Export cancelled.")
                return

            with open(path, 'w', newline='') as f:
                csv.writer(f).writerows([header, *rows])

            self._info(f"Spline knots exported to: {path}")

        except Exception as e:
            self._error(f"Export failed: {e}")
            traceback.print_exc()
