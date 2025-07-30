# napari-noise2vst

[![License MIT](https://img.shields.io/pypi/l/napari-noise2vst.svg?color=green)](https://github.com/IbrahimaAlain/napari-noise2vst/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-noise2vst.svg?color=green)](https://pypi.org/project/napari-noise2vst)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-noise2vst.svg?color=green)](https://python.org)
[![tests](https://github.com/IbrahimaAlain/napari-noise2vst/workflows/tests/badge.svg)](https://github.com/IbrahimaAlain/napari-noise2vst/actions)
[![codecov](https://codecov.io/gh/IbrahimaAlain/napari-noise2vst/branch/main/graph/badge.svg)](https://codecov.io/gh/IbrahimaAlain/napari-noise2vst)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-noise2vst)](https://napari-hub.org/plugins/napari-noise2vst)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A plugin for denoising microscopy images using Noise2VST

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->


## Installation

To install in an environment using conda:

```
conda env create -f environment.yml
conda activate noise2vst
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

To install latest development version :

```
pip install git+https://github.com/IbrahimaAlain/napari-noise2vst.git
```

## Dependencies

This plugin relies on the Noise2VST framework.
The exact version used is available at:
https://github.com/sherbret/Noise2VST/tree/feature/make-installable

    ✅ No manual installation is required — this version is installed automatically when you install the plugin.

## Citation

@article{herbreteau2024noise2vst,
  title={Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising},
  author={Herbreteau, S{\'e}bastien and Unser, Michael},
  journal={arXiv preprint arXiv:2407.17399},
  year={2024}
}

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
