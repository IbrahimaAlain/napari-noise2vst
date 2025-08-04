# napari-noise2vst

[![License MIT](https://img.shields.io/pypi/l/napari-noise2vst.svg?color=green)](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-noise2vst.svg?color=green)](https://pypi.org/project/napari-noise2vst)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-noise2vst.svg?color=green)](https://python.org)
[![tests](https://github.com/IbrahimaAlain/napari-noise2vst/workflows/tests/badge.svg)](https://github.com/IbrahimaAlain/napari-noise2vst/actions)
[![codecov](https://codecov.io/gh/IbrahimaAlain/napari-noise2vst/branch/main/graph/badge.svg)](https://codecov.io/gh/IbrahimaAlain/napari-noise2vst)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-noise2vst)](https://napari-hub.org/plugins/napari-noise2vst)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

> A plugin for denoising microscopy images using Noise2VST  
> Developed by **Ibrahima Alain GUEYE**

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#gett
Dependenciesing-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Dependencies

This plugin relies on the Noise2VST framework [S. Herbreteau and M. Unser, ICCV'25].
The source code is available at:
https://github.com/sherbret/Noise2VST
- ✅ No manual installation is required — this version is installed automatically when you install the plugin.

## Installation

To install in an environment using conda:

```
conda create --name napari-env
conda activate napari-env
conda install pip
```

You can install `napari-noise2vst` via [pip]:

```
pip install napari-noise2vst
```

If napari is not already installed, you can install `napari-noise2vst` with napari and Qt via:

```
pip install "napari-noise2vst[all]"
```

If you prefer installing napari separately:

```
pip install "napari[all]"
```

To install latest development version:

```
pip install git+https://github.com/IbrahimaAlain/napari-noise2vst.git
```

## Usage

After installation, you can launch the **Noise2VST Denoising** plugin directly from the napari interface.
In the napari top menu, go to:

**`Plugins > Noise2VST Denoising (Denoising Noise2VST)`**

![image_0.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_0.png)

Open your image by clicking:
**`File → Open File(s)...`**
Select the noisy image (e.g., .tif, .png, etc.) that you want to denoise. The image will appear in the napari viewer.

![image_1.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_1.png)

Once the image is loaded, scroll to the plugin panel on the right.
Set the number of training iterations using the slider (e.g., 2000).
Then click the Fit button to train the denoising model on the image.

The region shown here highlights the relevant settings and the training button.

![image_2.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_2.png)
![image_3.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_3.png)

A progress bar appears, indicating the training status in real time.
You can follow the advancement of model fitting visually.

![image_4.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_4.png)

Once training is complete, the plugin automatically stores the model weights.
Click the Run Denoising button to generate the denoised version of the input image.

![image_5.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_5.png)

The denoised image appears as a new layer in the napari viewer, alongside the original one.
You can toggle visibility, adjust contrast, and compare both layers interactively.

![image_6.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_6.png)

Click the Visualize VST button to display the spline transformation (VST) learned during training.
A matplotlib window pops up with a plot showing the input-output relationship.

![image_7.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_7.png)

To save the spline transformation values, click the Save Spline Knots button.
A dialog window opens to let you choose where to store the CSV file containing the knots.

![image_8.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/image_8.png)


## Citation

```BibTex
@article{herbreteau2024noise2vst,
  title={Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising},
  author={Herbreteau, S{\'e}bastien and Unser, Michael},
  journal={arXiv preprint arXiv:2407.17399},
  year={2024}
}
```

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/IbrahimaAlain/napari-noise2vst/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
