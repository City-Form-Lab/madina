# Installation

Currently, the Madina package is best used as stand-alone Python library codes with a `conda` environment, instead of being installed as a Python package with `pip` or `conda`. This guide will walk through the installation of the Madina package, its dependencies, as well as how to import the classes and methods.

## Installing Dependencies with `conda`

First, install `conda` on your computer - instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Then, in your command line (or the Anaconda/Miniconda prompt if you are using Windows), run the following commands to create a new conda environment called `una`, with all the required packages installed. If prompted to enter `y/n` during installation, enter `y` to proceed.

```bash
conda create --name una
conda install python=3.11
conda install -c conda-forge geopandas
conda install -c conda-forge tqdm
conda install numba
```

## Download Madina using Git

First, install `git` in your command line - instructions can be found [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Then, in your command line, clone the Madina package from Github to your local disk with the following command.

```bash
git clone https://github.com/City-Form-Lab/madina.git
```

## Importing the Madina Classes and Methods

To use the Madina package, we recommend importing the classes and methods from the Python files directly. To achieve this, you should first add the downloaded project directory to the Python system path. You should also run your project in the `una` environment created with `conda` just now, in order for dependencies of the project to be satisfied.

For example, suppose your own project in `/my_project/my_project.py` has the following file structure:

```
Madina
|-madina
| |-zonal
| | |-zonal.py
| |-una
|   |-betweenness.py
|-my_project
  |-my_project.py
```

Then, you should use the following lines for import:

```python
import sys
sys.path.append("..")

from madina.zonal.zonal import Zonal, Layer
from madina.una.betweenness import parallel_betweenness
```