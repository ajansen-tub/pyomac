.. pyomac documentation master file, created by
   sphinx-quickstart on Fri Jul  9 12:05:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyomac - Operational Modal Analysis of Civil Structures
=======================================================

`pyomac` aims to implement established methods for operational modal analysis (OMA).

.. note::

   This project is under active development.

Look how easy it is to use:

   import pyomac
   
   f, s, u = pyomac.fdd(data)

Features
--------

- Established OMA methods: FDD and SSI
- Plot basic result diagrams of these methods via `matplotlib`

Installation
------------

Install pyomac by running:

.. code-block:: shell

  $ python - m pip install pyomac


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   basic_usage
   ssi
   clustering
   api
   contributing


Contribute
----------

- Issue Tracker: github.com/ajansen-tub/pyomac/issues
- Source Code: github.com/ajansen-tub/pyomac

License
-------

The project is licensed under the MIT license.
