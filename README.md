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

## Usage

To begin, launch napari, then go to the top menu:
File → Open File...
Select the noisy image (e.g., .tif, .png, etc.) that you want to denoise. The image will appear in the napari viewer.

![img_1.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_1.png)

Once the image is loaded, scroll to the plugin panel on the right.
Set the number of training iterations using the slider (e.g., 2000).
Then click the Fit button to train the denoising model on the image.

The region shown here highlights the relevant settings and the training button.

![img_2.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_2.png)
![img_3.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_3.png)

A progress bar appears, indicating the training status in real time.
You can follow the advancement of model fitting visually.

![img_4.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_4.png)

Once training is complete, the plugin automatically stores the model weights.
Click the Run Denoise button to generate the denoised version of the input image.

![img_5.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_5.png)

The denoised image appears as a new layer in the napari viewer, alongside the original one.
You can toggle visibility, adjust contrast, and compare both layers interactively.

![img_6.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_6.png)

Click the Visualize SVT button to display the spline transformation (VST) learned during training.
A matplotlib window pops up with a plot showing the input-output relationship.

![img_7.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_7.png)

To save the spline transformation values, click the Save Spline Knots button.
A dialog window opens to let you choose where to store the CSV file containing the knots

![img_8.png](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/docs/images/img_8.png)


## Installation

To install in an environment using conda:

```
conda env create -f environment.yml
conda activate noise2vst
```

To install napari:

```
pip install "napari[all]"
```

To install latest development version :

```
pip install git+https://github.com/IbrahimaAlain/napari-noise2vst.git
```

## Dependencies

This plugin relies on the Noise2VST framework, developped by **Sébastien SHERBRET**.
The exact version used is available at:
https://github.com/sherbret/Noise2VST/tree/feature/make-installable

    ✅ No manual installation is required — this version is installed automatically when you install the plugin.

## Citation

```Bibtext
@article{herbreteau2024noise2vst,
  title={Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising},
  author={Herbreteau, S{\'e}bastien and Unser, Michael},
  journal={arXiv preprint arXiv:2407.17399},
  year={2024}
}
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-noise2vst" is free and open source software

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
