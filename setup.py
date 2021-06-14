'''
Text
'''

from distutils.core import setup

# files = ['resources/*']

requires = [
    'numpy',
    'scipy',
    'PeakUtils'
]

setup(name='pyoma',
      version='0.1',
      description='Tools for Operational Modal Analysis (OMA)',
      author='Andreas Jansen, Patrick Simon',
      author_email='',
      url='https://gitlab.com/ajansen1234/py_oma',
      packages=['pyoma'],
      install_requires=[
          'numpy (>=1.17.0)',
          'scipy',
          'PeakUtils'
      ]
      )
