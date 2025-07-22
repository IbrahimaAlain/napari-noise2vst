#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Noise2VST Napari Widget: Plugin de débruitage pour napari basé sur Noise2VST
"""

import os
import sys
import torch
import numpy as np
import traceback
from pathlib import Path
from skimage.util import img_as_float

from magicgui.widgets import Container, create_widget, PushButton, Label, Slider

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari

# Définir le chemin vers le dossier pretrained_weights dans le plugin
PLUGIN_DIR = Path(__file__).parent
WEIGHTS_DIR = PLUGIN_DIR / "pretrained_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# Ajouter le repo local noise2vst au sys.path
repo_path = PLUGIN_DIR / "noise2vst"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

# Imports du modèle et des utilitaires
from noise2vst.models.noise2vst import Noise2VST
from noise2vst.models.ffdnet import FFDNet
from noise2vst.models.drunet import DRUNet
from noise2vst.utilities.utilities import f_GAT, f_GAT_inv

# Import de la fonction de téléchargement des poids pré-entraînés
try:
    from napari_noise2vst.pretrained_weights.download import download_weights
except ImportError:
    download_weights = None


class Noise2VSTWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Noise2VST().to(self.device)

        # Widgets
        self.image_input = create_widget(label="Input Image", annotation="napari.layers.Image")
        self.train_button = PushButton(label="Train")
        self.eval_button = PushButton(label="Evaluate")
        self.status = Label(value="Status: Ready")
        self.iter_slider = Slider(
            value=2000,
            min=100,
            max=10000,
            step=100,
            description='Itérations :',
        )
        self.append(self.iter_slider)

        # Callbacks
        self.train_button.changed.connect(self.train_model)
        self.eval_button.changed.connect(self.evaluate_model)

        # Assemble widget
        self.extend([
            self.image_input,
            self.train_button,
            self.eval_button,
            self.status,
        ])

        # Télécharger les poids pré-entraînés au démarrage si possible
        if download_weights is not None:
            self._info("Checking pretrained weights...")
            try:
                download_weights()
                self._info("Pretrained weights ready.")
            except Exception as e:
                self._error(f"Automatic weight download failed: {e}")
        else:
            self._info("No automatic download function available.")

    def _info(self, msg):
        print(f"[INFO] {msg}")
        self.status.value = f"Status: {msg}"

    def _error(self, msg):
        print(f"[ERROR] {msg}")
        self.status.value = f"Error: {msg}"

    def _get_image_data(self):
        img_layer = self.image_input.value
        if img_layer is None:
            self._error("No image selected.")
            return None
        return img_as_float(img_layer.data)

    def load_models(self):
        ffdnet_path = WEIGHTS_DIR / "ffdnet_color.pth"
        drunet_path = WEIGHTS_DIR / "drunet_color.pth"

        ffdnet = FFDNet(color=True).to(self.device).eval()
        drunet = DRUNet(color=True).to(self.device).eval()

        ffdnet.load_state_dict(torch.load(ffdnet_path, map_location=self.device), strict=True)
        drunet.load_state_dict(torch.load(drunet_path, map_location=self.device), strict=True)

        return ffdnet, drunet
    

    def train_model(self, _=None):
        image = self._get_image_data()
        if image is None:
            return

        # Adapter l'image au format (N, C, H, W)
        if image.ndim == 2:  # grayscale
            image = image[None, None, :, :]
        elif image.ndim == 3:  # color
            image = image.transpose(2, 0, 1)[None, :]
        else:
            self._error(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        # Télécharger les poids (sécurité)
        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self._error(f"Download failed: {e}")
                return

        try:
            ffdnet, _ = self.load_models()
        except Exception as e:
            self._error(f"Model loading failed: {e}")
            return

        # Chargement des poids spline si disponibles
        spline_path = WEIGHTS_DIR / "noise2vst_spline.pth"
        if spline_path.exists():
            try:
                self.model.load_state_dict(torch.load(spline_path, map_location=self.device))
                self._info("Spline weights loaded.")
            except Exception as e:
                self._error(f"Failed to load spline weights: {e}")

        # Entraînement
        try:
            nb_iter = self.iter_slider.value()
            self.model.fit(image, ffdnet, nb_iterations=nb_iter)

            self._info("Training complete.")
        except Exception as e:
            self._error(f"Training failed: {e}")
            traceback.print_exc()
            return

        # Sauvegarde des poids spline
        try:
            torch.save(self.model.state_dict(), spline_path)
            self._info(f"Weights saved to {spline_path}")
        except Exception as e:
            self._error(f"Failed to save weights: {e}")

    def evaluate_model(self, _=None):
        image = self._get_image_data()
        if image is None:
            return

        # Adapter l'image au format (N, C, H, W)
        if image.ndim == 2:
            image = image[None, None, :, :]
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)[None, :]
        else:
            self._error(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        # Télécharger les poids (sécurité)
        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self._error(f"Download failed: {e}")
                return

        try:
            _, drunet = self.load_models()
        except Exception as e:
            self._error(f"Model loading failed: {e}")
            return

        try:
            with torch.no_grad():
                output = self.model(image, drunet)
                output = output.cpu().numpy()
        except Exception as e:
            self._error(f"Inference failed: {e}")
            traceback.print_exc()
            return

        name = self.image_input.value.name + "_denoised"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = output
        else:
            self.viewer.add_image(output, name=name)
        self._info("Denoising complete.")
