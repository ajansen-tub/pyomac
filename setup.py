from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='pyoma',
      packages=['pyoma'],
      version='0.1',
      license='MIT',
      author='Andreas Jansen, Patrick Simon',
      author_email='andreas.jansen@tu-berlin.de',
      description='Tools for Operational Modal Analysis (OMA)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ajansen-tub/pyoma',
      download_url='https://github.com/ajansen-tub/pyoma/v_01.tar.gz',
      keywords=['Operational Modal Analysis', 'Structural Dynamics', 'Frequency Domain Decomposition',
                'Stochastic Subspace Identification'],
      install_requires=[
          'numpy',
          'scipy',
          'PeakUtils',
          'matplotlib'
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Structural Engineers',  # Define that your audience are developers
          'Topic :: Operational Modal Analysis',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
      )
