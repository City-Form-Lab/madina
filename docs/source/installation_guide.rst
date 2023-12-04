Installation Guide
==================

1) Get Conda
--------------
Download and install conda from here: `Installing Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_. Once you're done, continute to the next section.

Conda is a an environment and packaee manager that simplify the installation of python packages. Geopandas, a python package that is used inside Madina recmment using conda as the simplest and most reliable way to install that package. a python environment is a collection of packages with specific version numbers. Conda makes it easy to create and manage environments.


2) Install Madina
-----------------------
Once you have installed conda, run these commandsm in your terminal (command prompt or powershell in Windows for example). run the following command to create a new python environment called "madina_env"

.. code-block:: console

   conda create --name=madina_env python

Avctivate the newly created environment by running this command:

.. code-block:: console

    conda activate madina_env

Madina uses a library called geopandas, the next three commands are to `install geopandas according to their official guide <https://geopandas.org/en/stable/getting_started/install.html>`_ . The next command adds the channel "conda-forge" as a source to install packages in the activated environment "madina_env":

.. code-block:: console

    conda config --env --add channels conda-forge

The next command ensures the channel priority

.. code-block:: console

    conda config --env --set channel_priority strict

The next command installs geopandas

.. code-block:: console

    conda install python=3 geopandas pyogrio

The next cxommand installs pyogrio, a dependency

.. code-block:: console

    conda install pyogrio

The next command installs madina

.. code-block:: console

    pip install -i https://test.pypi.org/simple/ madina




3) Run an example
If you're familiar with programming in python, feel free to choose your favorate text editor. The folloing instructions are useful to familiarize yourself with running madina for the first time.
