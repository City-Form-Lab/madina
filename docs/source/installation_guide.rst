Installation Guide
==================

1) Get Conda
--------------
Download and install conda from here: `Installing Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_.
Once you're done, continute to the next section.

Conda is a an environment and packaee manager that simplify the installation of python packages.
A python environment is a collection of packages in speccific versions that work well together for a specific project. 
Conda makes it easy to create and manage environments, and if you're using python for multiple projects, it might be helpful to create an environment for each of those projects to make sure each project is using the recommended version of python and the recommended versions of each package that progect depends on.

2) Install Madina
-----------------------
Windows, MacOS, Linux
^^^^^^^^^^^^^^^^^^^^^^^
Madina depends on a python package called  `geopandas <geopandas.org/en/stable/>`_, which recommends using conda as the simplest and most reliable way to install that package.
Once you have installed conda, run these commandsm in your terminal (command prompt or powershell in Windows for example) as an administrator. run the following command to create a new python environment called "madina_env"

.. code-block:: console

   conda create --name=madina_env python

Avctivate the newly created environment by running this command:

.. code-block:: console

    conda activate madina_env

The next three commands are to `install geopandas according to their official guide <https://geopandas.org/en/stable/getting_started/install.html>`_ . The next command adds the channel "conda-forge" as a source to install packages in the activated environment "madina_env":

.. code-block:: console

    conda config --env --add channels conda-forge

The next command ensures the channel priority

.. code-block:: console

    conda config --env --set channel_priority strict

The next command installs geopandas

.. code-block:: console

    conda install python=3 geopandas pyogrio

The next command installs madina

.. code-block:: console

    pip install -i https://test.pypi.org/simple/ madina==0.0.7

3) Run an example
-----------------------

If you're familiar with programming in python, feel free to choose your favorate text editor.
The folloing instructions are useful to familiarize yourself with running madina for the first time.

* Download `Visual Studio Code <https://code.visualstudio.com/>`_, a simple and powerful code editor. 
* Download `The example folder <https://www.dropbox.com/scl/fi/1fbidbc5bqz7ccn61u1yq/examples.zip?rlkey=y0ppgukbyck0scw6pakrcn7f5&dl=1>`_, and unzip it.
* In VS code, Click FIle > Open FOlder and navigate to the unzipped "examples folder"
* Make sure to set the environment to "madina_env" that you created during package installation.
* Open the example notebooks and follow along step by step
* if you want to take a quick look, these python notebooks are available in rthe example section of this documentation


