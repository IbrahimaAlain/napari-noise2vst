"""
This module defines a Noise2VST denoising widget for Napari,
including automatic download and loading of pretrained models.
"""

import sys
import os
import torch
import numpy as np
import urllib.request
from pathlib import Path

from magicgui.widgets import Container, create_widget, FileEdit, PushButton
from skimage.util import img_as_float

# Import du module noise2vst installé via pip
import noise2vst
from noise2vst.models.noise2vst import Noise2VST
from noise2vst.models.ffdnet import FFDNet
from noise2vst.models.drunet import DRUNet
from noise2vst.utilities.utilities import f_GAT, f_GAT_inv

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari

# Détection du dossier d'installation de noise2vst
NOISE2VST_PATH = os.path.dirname(noise2vst.__file__)
WEIGHTS_DIR = os.path.join(NOISE2VST_PATH, "pretrained_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class Noise2VSTWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Noise2VST().to(self.device)

        # Widgets
        self.image_input = create_widget(label="Input Image", annotation="napari.layers.Image")
        self.spline_weights_path = FileEdit(label="Load weights (.pth)", mode="r", filter="*.pth")
        self.save_weights_path = FileEdit(label="Save weights to", mode="w", filter="*.pth")
        self.train_button = PushButton(text="Train")
        self.eval_button = PushButton(text="Evaluate")

        # Callbacks
        self.train_button.clicked.connect(self.train_model)
        self.eval_button.clicked.connect(self.evaluate_model)

        # Assemble widget
        self.extend([
            self.image_input,
            self.spline_weights_path,
            self.save_weights_path,
            self.train_button,
            self.eval_button,
        ])

    def _info(self, msg):
        print(f"[INFO] {msg}")

    def _error(self, msg):
        print(f"[ERROR] {msg}")

    def _get_image_data(self):
        img_layer = self.image_input.value
        if img_layer is None:
            self._error("No image selected.")
            return None
        return img_as_float(img_layer.data)

    def download_weights(self):
        base_url = "https://github.com/cszn/KAIR/releases/download/v1.0/"
        filenames = ["ffdnet_color.pth", "drunet_color.pth"]

        for fname in filenames:
            fpath = os.path.join(WEIGHTS_DIR, fname)
            if not os.path.exists(fpath):
                self._info(f"Téléchargement de {fname}...")
                try:
                    urllib.request.urlretrieve(base_url + fname, fpath)
                    self._info(f"{fname} téléchargé avec succès.")
                except Exception as e:
                    self._error(f"Échec du téléchargement de {fname}: {e}")

    def load_models(self):
        ffdnet = FFDNet(color=True).to(self.device).eval()
        drunet = DRUNet(color=True).to(self.device).eval()

        ffdnet.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "ffdnet_color.pth"), map_location=self.device), strict=True)
        drunet.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "drunet_color.pth"), map_location=self.device), strict=True)

        return ffdnet, drunet

    def train_model(self):
        image = self._get_image_data()
        if image is None:
            return

        self.download_weights()
        try:
            ffdnet, _ = self.load_models()
        except Exception as e:
            self._error(f"Erreur lors du chargement des modèles : {e}")
            return

        if self.spline_weights_path.value:
            try:
                self.model.load_state_dict(torch.load(self.spline_weights_path.value, map_location=self.device))
                self._info("Spline weights loaded.")
            except Exception as e:
                self._error(f"Could not load weights: {e}")

        try:
            self.model.fit(image, ffdnet, nb_iterations=2000)
            self._info("Training complete.")
        except Exception as e:
            self._error(f"Training failed: {e}")
            return

        if self.save_weights_path.value:
            try:
                torch.save(self.model.state_dict(), self.save_weights_path.value)
                self._info(f"Weights saved to {self.save_weights_path.value}")
            except Exception as e:
                self._error(f"Failed to save weights: {e}")

    def evaluate_model(self):
        image = self._get_image_data()
        if image is None:
            return

        try:
            _, drunet = self.load_models()
        except Exception as e:
            self._error(f"Erreur lors du chargement de DRUNet : {e}")
            return

        try:
            with torch.no_grad():
                output = self.model(image, drunet)
                output = output.cpu().numpy()
        except Exception as e:
            self._error(f"Inference failed: {e}")
            return

        name = self.image_input.value.name + "_denoised"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = output
        else:
            self.viewer.add_image(output, name=name)
