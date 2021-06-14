from distutils.core import setup
from os import path
from io import open


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(name='pyomac',
      packages=['pyomac'],
      version='0.1.1',
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
      install_requires=requirements,
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
