Installation Guide
==================

1) Get Conda
--------------




2) Install Madina
-----------------------
Once you have installed conda, run these commandsm in your terminal, run the following command to create a new python environment called "madina_env"

.. code-block:: console

   conda create --name=madina_env python

Avctivate the newly created environment by running this command:

.. code-block:: console

    conda activate madina_env

Madina uses a library called geopandas, the next three commands are to install geopandas according to their official guide. The next command adds the channel "conda-forge" as a source to install packages in the activated environment "madina_env":

.. code-block:: console

    conda config --env --add channels conda-forge

The next command ensures the channel priority 

.. code-block:: console

    conda config --env --set channel_priority strict

The next command installs geopandas 

.. code-block:: console

    conda install python=3 geopandas

The next cxommand installs pyogrio, a dependency

.. code-block:: console

    conda install pyogrio

The next command installs madina

.. code-block:: console

    pip install -i https://test.pypi.org/simple/ madina==0.0.3

