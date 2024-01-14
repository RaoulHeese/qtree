"""Data for quantum decision trees.
"""
from abc import ABC, abstractmethod
import numpy as np
import os


class BDS(ABC):  # Binary data set
    """Abstract base class for binary data sets.
    
    Parameters
    ----------
    
    **kwargs : Arguments for data loader.
    """

    def __init__(self, **kwargs):
        self._load(**kwargs)
        self._X = BDS._ensure_array(self._X).astype(bool)
        self._y = BDS._ensure_array(self._y).astype(bool)

    @property
    def X(self):
        """numpy.ndarray of shape (n_samples,n_features) : Features.
        """
        return self._X

    @property
    def y(self):
        """numpy.ndarray of shape (n_samples,n_labels) : Labels.
        """
        return self._y

    @staticmethod
    def _ensure_array(a):
        a = np.asarray(a)
        if len(a.shape) == 1:
            a = a.reshape(-1, 1)
        return a

    def __str__(self):
        return f"X: {self.X.shape}, y: {self.y.shape}"

    @abstractmethod
    def _load(self, **kwargs):
        raise NotImplementedError


class BDS_test(BDS):
    """Binary data set: XOR test data set consisting of 7 features and 1 label.
    
    Parameters
    ----------
    
    **kwargs :
        ``N`` (``int``) determines the number of data points, defaults to 5.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self, N=5):
        x1 = [[1, 0, 0, 0, 1] + [0] * (N - 5)]
        x2 = [[1, 1, 0, 0, 1] + [0] * (N - 5)]
        x3 = [[0, 1, 1, 0, 1] + [0] * (N - 5)]
        x4 = [[0 for n in range(N)]]
        x5 = [[0 for n in range(N)]]
        x6 = [[0 for n in range(N)]]
        x7 = [[0 for n in range(N)]]
        y1 = [[(a + b + c) % 2 for a, b, c in zip(x1[0], x2[0], x3[0])]]
        X = np.concatenate((x1, x2, x3, x4, x5, x6, x7), axis=0).T
        y = np.concatenate((y1), axis=0).T
        self._X = np.asarray(X)
        self._y = np.asarray(y).reshape(-1, 1)


class BDS_tic_tac_toe(BDS):
    """Binary data set:  Binarized version of the Tic-Tac-Toe Endgame Data Set.
    Each 15-dimensional feature vector represents a Tic-Tac-Toe playfield, the 
    corresponding label represents the winning player.
    
        Original source: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
        
        Original creator: David W. Aha (aha@cs.jhu.edu)
        
        Note: Encoding to binary and decoding from binary is realized via 
        ``BDS_tic_tac_toe._encode_x`` and 
        ``BDS_tic_tac_toe._decode_x``, respectively.
    
    Parameters
    ----------
    
    root : str
        Root directory to load data file ``data/tictactoe/tic-tac-toe.npz``.
    
    **kwargs :
        Unused.
    """

    def __init__(self, root="", **kwargs):
        self._root = root
        super().__init__(**kwargs)

    @staticmethod
    def _encode_x(x, N=15):  # ternary -> int -> binary

        def x_to_int(x):
            i = 0
            for idx, val in enumerate(x[::-1]):
                i += val * 3 ** idx
            return i

        def int_to_b(i, N):
            b = [0 for _ in range(N)]
            bin_str = bin(i)[2:]  # bit string
            for idx, val in enumerate(bin_str):
                b[N - len(bin_str) + idx] = int(val)
            # b = b[::-1]
            return b

        def convert_x(x):
            return int_to_b(x_to_int(x), N)

        x = np.array(x).ravel()
        assert x.size == 9
        return convert_x(x)

    @staticmethod
    def _decode_x(x, N=9):  # binary -> int -> ternary

        def x_to_int(x):
            i = 0
            for idx, val in enumerate(x[::-1]):
                i += val * 2 ** idx
            return i

        def int_to_t(i, N):
            t = [0 for _ in range(N)]
            ter_str = np.base_repr(i, base=3)
            for idx, val in enumerate(ter_str):
                t[N - len(ter_str) + idx] = int(val)
            # t = t[::-1]
            return t

        def convert_x(x):
            return int_to_t(x_to_int(x), N)

        x = np.array(x).ravel()
        assert x.size == 15
        return convert_x(x)

    @staticmethod
    def print_playfield(x):
        """Print playfield represented by feature vector ``x``.
        """

        x = np.asarray(x.copy()).ravel()
        if x.size == 9:
            pass
        elif x.size == 15:
            x = BDS_tic_tac_toe._decode_x(x)
        else:
            raise ValueError

        x = np.asarray(x, dtype=object).ravel()
        x[x == 0] = 'X'  # x
        x[x == 1] = 'O'  # o
        x[x == 2] = ' '  # b

        print("_____")
        print("|{}{}{}|".format(x[0], x[1], x[2]))
        print("|{}{}{}|".format(x[3], x[4], x[5]))
        print("|{}{}{}|".format(x[6], x[7], x[8]))
        print("`````")

    def _load(self, **kwargs):
        data_file = os.path.join(self._root, "data/tictactoe/tic-tac-toe.npz")
        data = np.load(data_file)
        self._X = data['X']
        self._y = data['y']
