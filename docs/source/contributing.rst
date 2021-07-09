

Building the documentation
--------------------------

.. code-block:: shell

$ python - m pip install pyomac



Building the documentation
--------------------------


# from:
# https://stackoverflow.com/questions/37891550/jupyter-notebook-running-kernel-in-different-env
# https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments

# needed:
.. code-block:: shell

  $ python -m ipykernel install --user --name pyomac_venv --display "Pyomac VENV"



  .. code-block:: shell

  $ python -m jupyter notebook

  
  .. code-block:: shell

  $ python -m jupyter nbconvert ./docs/source/basic_usage.ipynb --to rst
