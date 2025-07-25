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
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.util import img_as_float

from magicgui.widgets import Container, create_widget, PushButton, Label, Slider, ProgressBar, FileEdit

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

        # === Input ===
        self.input_label = Label(value="Input Image:")
        self.image_input = create_widget(label="Input Image", annotation="napari.layers.Image")

        # === Step 1: Training ===
        self.step1_label = Label(value="Step 1: Train")
        self.iter_slider = Slider(
            label="Number of training iterations:",
            value=2000,
            min=100,
            max=5000,
            step=100,
        )
        self.train_button = PushButton(label="Fit the VST model")
        self.progress_bar = ProgressBar(min=0, max=100, label = "loading", visible=False)

        # Container Step 1
        self.step1_container = Container(widgets=[
            self.step1_label,
            self.iter_slider,
            self.train_button,
            self.progress_bar,
        ])

        # === Step 2: Predict & Evaluate ===
        self.step2_label = Label(value="Step 2: Predict & Evaluate")
        self.eval_button = PushButton(label="Run Denoising")
        self.plot_spline_button = PushButton(label="Visualize VST")
        self.save_spline_button = PushButton(label="Save Spline Knots")

        # Container Step 2
        self.step2_container = Container(widgets=[
            self.step2_label,
            self.eval_button,
            self.plot_spline_button,
            self.save_spline_button,
        ])
        self.step2_container.visible = False

        self.status = Label(value="Status: Ready")

        # === Callbacks ===
        self.train_button.changed.connect(self.train_model)
        self.eval_button.changed.connect(self.evaluate_model)
        self.plot_spline_button.changed.connect(self.plot_spline)
        self.save_spline_button.changed.connect(self.export_spline_knots)

        # === Assemble all ===
        self.extend([
            self.input_label,
            self.image_input,
            self.step1_container,
            self.step2_container,
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

    def load_models(self, image: torch.Tensor):
        """
        Charge dynamiquement les modèles FFDNet et DRUNet selon le nombre de canaux de l'image.
        """
        is_color = image.shape[1] == 3

        if is_color:
            ffdnet_path = WEIGHTS_DIR / "ffdnet_color.pth"
            drunet_path = WEIGHTS_DIR / "drunet_color.pth"
        else:
            ffdnet_path = WEIGHTS_DIR / "ffdnet_gray.pth"
            drunet_path = WEIGHTS_DIR / "drunet_gray.pth"

        ffdnet = FFDNet(color=is_color).to(self.device).eval()
        drunet = DRUNet(color=is_color).to(self.device).eval()
 
        ffdnet.load_state_dict(torch.load(ffdnet_path, map_location=self.device), strict=True)
        drunet.load_state_dict(torch.load(drunet_path, map_location=self.device), strict=True)

        return ffdnet, drunet

        
    def train_model(self, _=None):
        image = self._get_image_data()
        if image is None:
            return

        if image.ndim == 2:
            image = image[None, None, :, :]
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)[None, :]
        elif image.ndim == 4:
            pass
        else:
            self._error(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self._error(f"Download failed: {e}")
                return

        try:
            ffdnet, _ = self.load_models(image)
        except Exception as e:
            self._error(f"Model loading failed: {e}")
            return

        spline_path = WEIGHTS_DIR / "noise2vst_spline.pth"
        if spline_path.exists():
            try:
                self.model.load_state_dict(torch.load(spline_path, map_location=self.device))
                self._info("Spline weights loaded.")
            except Exception as e:
                self._error(f"Failed to load spline weights: {e}")

        try:
            self.progress_bar.visible = True
            self.progress_bar.value = 0
            nb_iter = self.iter_slider.value
            self.model.fit(image, ffdnet, nb_iterations=nb_iter, progress_callback=lambda v: setattr(self.progress_bar, "value", v))
            self._info("Training complete.")
            self.progress_bar.visible = False
            self.step2_container.visible = True
        except Exception as e:
            self._error(f"Training failed: {e}")
            traceback.print_exc()
            return

        try:
            torch.save(self.model.state_dict(), spline_path)
            self._info(f"Weights saved to {spline_path}")
        except Exception as e:
            self._error(f"Failed to save weights: {e}")

    def evaluate_model(self, _=None):
        image = self._get_image_data()
        if image is None:
            return

        if image.ndim == 2:
            image = image[None, None, :, :]
        elif image.ndim == 3: 
            image = image.transpose(2, 0, 1)[None, :]
        elif image.ndim == 4:
            pass
        else:
            self._error(f"Unsupported image shape: {image.shape}")
            return

        image = torch.from_numpy(image).float().to(self.device)

        if download_weights is not None:
            try:
                download_weights()
            except Exception as e:
                self._error(f"Download failed: {e}")
                return

        try:
            _, drunet = self.load_models(image)
        except Exception as e:
            self._error(f"Model loading failed: {e}")
            return

        try:
            with torch.no_grad():
                output = self.model(image, drunet)
                if output.dim() == 4 and output.shape[0] == 1:
                    output = output.squeeze(0)
                output = output.permute(1, 2, 0).cpu().numpy()

                if output.shape[2] == 1:
                    output = output[..., 0]
                    rgb_flag = False
                else:
                    rgb_flag = True
        except Exception as e:
            self._error(f"Inference failed: {e}")
            traceback.print_exc()
            return

        name = self.image_input.value.name + "_denoised"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = output
        else:
            self.viewer.add_image(output, name=name, rgb=rgb_flag)
        self._info("Denoising complete.")

    def plot_spline(self, _=None):
        try:
            spline_path = WEIGHTS_DIR / "noise2vst_spline.pth"
            if not spline_path.exists():
                self._error("Les poids spline n'existent pas encore.")
                return

            self.model.load_state_dict(torch.load(spline_path, map_location=self.device))

            spline1 = self.model.spline1
            spline2 = self.model.spline2

            x = torch.linspace(0, 1, 1000).to(self.device)
            y1 = spline1(x).detach().cpu().numpy()
            y2 = spline2(x).detach().cpu().numpy()

            plt.figure(figsize=(8, 4))
            plt.plot(x.cpu().numpy(), y1, label="Spline 1 (VST)", color='blue')
            plt.plot(x.cpu().numpy(), y2, label="Spline 2 (Inverse VST)", color='orange')
            plt.title("Courbes des splines du VST")
            plt.xlabel("Entrée")
            plt.ylabel("Sortie")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            self._info("Affichage des splines réussi.")
        except Exception as e:
            self._error(f"Erreur lors de l'affichage des splines : {e}")
            traceback.print_exc()
                
    def export_spline_knots(self, _=None):
        spline_path = WEIGHTS_DIR / "noise2vst_spline.pth"

        if not spline_path.exists():
            self._error("Les poids de la VST n'ont pas été trouvés. Veuillez entraîner le modèle d'abord.")
            return

        try:
            state_dict = torch.load(spline_path, map_location=self.device)
            theta_in = state_dict["spline1.theta"].cpu().numpy()
            theta_out = state_dict["spline2.theta"].cpu().numpy()

            # On suppose que chaque theta correspond à une spline cubique avec 10 noeuds (par ex.)
            x = np.linspace(0, 1, len(theta_in))
            knots = list(zip(x, theta_in, theta_out))

            path = FileDialog(mode='w', filter='*.csv', label='Save Spline Knots').get_path()
            if not path:
                return

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y_in', 'y_out'])
                writer.writerows(knots)

            self._info(f"Nœuds de la spline exportés vers : {path}")
        except Exception as e:
            self._error(f"Erreur lors de l’export : {e}")
            traceback.print_exc()
