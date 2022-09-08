******************************
Quantum Decision Trees (qtree)
******************************

.. image:: https://github.com/RaoulHeese/qtree/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/RaoulHeese/qtree/actions/workflows/tests.yml
    :alt: GitHub Actions
	
.. image:: https://readthedocs.org/projects/qtree/badge/?version=latest
    :target: https://qtree.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status	
	
.. image:: https://img.shields.io/badge/license-MIT-lightgrey
    :target: https://github.com/RaoulHeese/qtree/blob/main/LICENSE
    :alt: MIT License	
	
This Python package implements quantum decision tree classifiers for binary data. The details of the method can be found in `Representation of binary classification trees with binary features by quantum circuits <https://doi.org/10.22331/q-2022-03-30-676>`_

**Installation**

Install via ``pip`` or clone this repository. In order to use ``pip``, type:

.. code-block:: sh

    $ pip install quantum-tree
	
**Usage**

Minimal working example:

.. code-block:: python

  # create quantum tree instance
  from qtree.qtree import QTree
  qtree = QTree(max_depth=1)

  # create simple training data
  import numpy as np
  X = np.array([[1,0,0], [0,1,0], [0,0,1]]) # features
  y = np.array([[0,0], [0,1], [1,1]])       # labels
  
  # fit quantum tree
  qtree.fit(X, y)

  # make quantum tree prediction
  qtree.predict([[0,0,1]])
  
**Documentation**

Documentation is available on `<https://qtree.readthedocs.io/en/latest>`_.

Demo notebooks can be found in the ``examples/`` directory.

📖 **Citation**

If you find this code useful in your research, please consider citing:

.. code-block:: tex

    @article{Heese2022representationof,
             doi = {10.22331/q-2022-03-30-676},
             url = {https://doi.org/10.22331/q-2022-03-30-676},
             title = {Representation of binary classification trees with binary features by quantum circuits},
             author = {Heese, Raoul and Bickert, Patricia and Niederle, Astrid Elisa},
             journal = {{Quantum}},
             issn = {2521-327X},
             publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
             volume = {6},
             pages = {676},
             month = {3},
             year = {2022}
            }
