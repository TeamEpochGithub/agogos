<p align="center">
    <img src="./docs/source/_static/agogos.png" width="200">
</p>

<h1 align="center">Agogos</h1>

[![PyPI Latest Release](https://img.shields.io/pypi/v/agogos.svg)](https://pypi.org/project/agogos/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/agogos.svg?label=PyPI%20downloads)](https://pypi.org/project/agogos/)

[WARNING] This project has been moved into [EpochLib](https://github.com/TeamEpochGithub/epochlib) under pipeline and is not updated anymore. This package contains modules and classes necessary to construct an ml pipeline for machine learning competitions.

## Description

<p align="center">
    <img src="./docs/images/Screenshot_2024-03-27-22-01-49_5760x1080.png" width="300">
</p>

## Pytest coverage report

To generate pytest coverage report run

```shell
python -m pytest --cov=agogos --cov-report=html:coverage_re
```

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in the browser:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
