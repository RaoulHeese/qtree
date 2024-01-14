"""Tests: Quantum trees.
"""
import pytest
import numpy as np
from qtree.qtree import QTree


def test_mwe():
    """Minimal working example from readme.
    """
    qtree = QTree(max_depth=1)
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = np.array([[0, 0], [0, 1], [1, 1]])
    qtree.fit(X, y)
    qtree.predict([[0, 0, 1]])
