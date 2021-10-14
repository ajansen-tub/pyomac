from distutils.core import setup
from os import path
from io import open

from setuptools import find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='pyomac',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      version='0.1.4',
      license='MIT',
      author='Andreas Jansen, Patrick Simon',
      author_email='andreas.jansen@tu-berlin.de',
      description='Tools for Operational Modal Analysis (OMA)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ajansen-tub/pyomac',
      download_url='https://github.com/ajansen-tub/pyomac/v_01.tar.gz',
      keywords=['Operational Modal Analysis', 'Structural Dynamics', 'Frequency Domain Decomposition',
                'Stochastic Subspace Identification'],
      install_requires=[
          'numpy',
          'scipy',
          'PeakUtils',
          'matplotlib',
          'scikit-learn'
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
      )
