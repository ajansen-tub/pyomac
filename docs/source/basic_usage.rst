Basic Usage
===========

.. code:: ipython3

    
    # The `pyomac` package is not installed in the development environment.
    # In order for this notebook to work, the root of the project has to be added to the sys.path:
    # This has to be deleted in the resulting *.rst documentation file
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../../'))
    

.. code:: ipython3

    import numpy as np
    import pyomac


::


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-2-1429d3864bc7> in <module>
          1 import numpy as np
    ----> 2 import pyomac
    

    ~\Documents\Code\pyomac\pyomac\__init__.py in <module>
    ----> 1 from pyomac.fdd import fdd, fft, peak_picking
          2 from pyomac.ssi import ssi_cov
    

    ~\Documents\Code\pyomac\pyomac\fdd.py in <module>
          1 import numpy as np
          2 from scipy import signal
    ----> 3 from peakutils import indexes
          4 
          5 
    

    ModuleNotFoundError: No module named 'peakutils'


