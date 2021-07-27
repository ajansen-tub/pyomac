Contributing
============


From the project root run:

.. code-block:: shell

  $ python -m pip install -e .


Building the documentation locally
----------------------------------

.. code-block:: shell

  $ python -m pip install -r requirements_dev.txt


  
.. code-block:: shell

  $ cd docs

  $ ./make html




Building the documentation from Jupyter notebooks
-------------------------------------------------


# from:
# https://stackoverflow.com/questions/37891550/jupyter-notebook-running-kernel-in-different-env
# https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments

# needed:
.. code-block:: shell

  $ python -m ipykernel install --user --name pyomac_venv --display "Pyomac VENV"



  .. code-block:: shell

  $ python -m jupyter notebook

  .. note::
     This workflow needs `pandoc` as a dependency. Probably a hurdle.
     https://nbconvert.readthedocs.io/en/latest/usage.html#restructuredtext

     One solution is to use [`nbsphinx`](https://nbsphinx.readthedocs.io/en/0.8.6/).

  .. code-block:: shell

  $ python -m jupyter nbconvert ./docs/source/basic_usage.ipynb --to rst
