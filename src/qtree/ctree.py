"""Classical decision trees.
"""
import numpy as np
from itertools import product
from sklearn import metrics
import warnings
import copy
import logging


class Node(object):
    """Classical decision tree node (root, inner node or leaf).
    
    Parameters
    ----------
    
    depth : int
        Depth of this node. The root corresponds to ``depth=0``.
        
    y_dim : int
        Number of binary labels.
   
    default_y : int or None
        Default value based on majority of classes in training data. Is used 
        if a classification is ambiguous in a leaf. Set to None to disable.
        
    logger : logging.Logger or None
        Logger for all output purposes. If None, print to std is used.
    """

    def __init__(self, depth, y_dim, default_y=None, logger=None):
        self._depth = depth
        self._y_dim = y_dim
        self._split_index = None
        self._split_0_node = None
        self._split_1_node = None
        self._proba_prediction = None
        self._undefined_prediction = None
        self._default_y = default_y  # if no samples are available: predict class with the highest occurence in training
        self._logger = logger
        self._warning = None  # only store last warning

    @property
    def py_default(self):
        if self._default_y is None:
            py = [[.5] * self._y_dim, [.5] * self._y_dim]
        else:
            py = np.empty((2, self._y_dim))
            for yi in range(self._y_dim):
                if self._default_y[yi] is None:
                    py[:, yi] = [.5, .5]
                elif self._default_y[yi] == 0:
                    py[:, yi] = [1., 0.]
                elif self._default_y[yi] == 1:
                    py[:, yi] = [0., 1.]
                else:
                    raise ValueError
        return py

    def __str__(self):
        return self._to_str()

    def _print(self, msg, lvl=logging.DEBUG):
        if self._logger is not None:
            self._logger.log(lvl, msg)
        else:
            print(msg)

    def _push_warning(self, msg, omit_warnings):
        self._warning = msg
        if not omit_warnings:
            self._print("Warning: {}".format(msg), logging.WARNING)

    def _split(self, tree_dict):
        if len(tree_dict) > 0 and type(list(tree_dict.values())[0]) is list and np.all(
                [type(split) is dict for split in list(tree_dict.values())[0]]):
            self._split_index = list(tree_dict.keys())[0]
            split_list = list(tree_dict.values())[0]
            self._split_0_node = Node(self._depth + 1, self._y_dim, self._default_y, self._logger)
            self._split_0_node._split(split_list[0])
            self._split_1_node = Node(self._depth + 1, self._y_dim, self._default_y, self._logger)
            self._split_1_node._split(split_list[1])
        else:
            self._leaf(tree_dict)

    def _leaf(self, tree_dict):
        if len(tree_dict) == 0 or 0 not in tree_dict or 1 not in tree_dict:
            self._proba_prediction = self.py_default
            self._undefined_prediction = True
        else:
            py = [tree_dict[0], tree_dict[1]]
            self._proba_prediction = py
            self._undefined_prediction = False

    def _to_str(self, indent=0, tab="\t", title=""):
        out = ""
        if len(title) > 0:
            out += "{}{}".format(tab * indent, title) + "\n"
        if self._split_0_node is not None:
            out += "{}x[{}] = {}".format(tab * indent, self._split_index, 0) + "\n"
            out += self._split_0_node._to_str(indent + 1, tab)
        if self._split_1_node is not None:
            out += "{}x[{}] = {}".format(tab * indent, self._split_index, 1) + "\n"
            out += self._split_1_node._to_str(indent + 1, tab)
        if self._proba_prediction is not None:
            if self._default_y is None:
                undef_tag = "?"
            else:
                undef_tag = "! y=[{}]".format(
                    ",".join(["{:d}".format(y) if y is not None else "?" for y in self._default_y]))
            y_predict = self._predict(None, True)
            y_proba_0 = self._proba_prediction[0]
            y_proba_1 = self._proba_prediction[1]
            tag_str = (" " + undef_tag) if self._undefined_prediction else ""
            for idx in range(self._y_dim):
                out += "{}{}".format(tab * indent,
                                     "y_{} = {} ({:.3f}/{:.3f}){}".format(idx, y_predict[idx], y_proba_0[idx],
                                                                          y_proba_1[idx], tag_str)) + "\n"
        return out

    def _predict(self, x=None, omit_warnings=False, verbose=False):
        return np.argmax(self._predict_proba(x, omit_warnings, verbose), axis=0)

    def _predict_proba(self, x=None, omit_warnings=False,
                       verbose=False):  # verbose = True: print path through tree for each prediction
        if verbose:
            self._print("predict_proba: depth={}, split_index={}".format(self._depth, self._split_index))
        if self._proba_prediction is not None:
            if verbose:
                self._print("\tpredict_proba: prediction={}{}".format(self._proba_prediction,
                                                                      " (undefined)" if self._undefined_prediction else ""))
            if self._undefined_prediction:
                self._push_warning("Undefined prediction for x={}!".format(x), omit_warnings)
            return self._proba_prediction  # [[y0=0,y1=0,...],[y0=1,y1=1,...]]
        if x[self._split_index] == 0:
            if verbose:
                self._print("\tpredict_proba: x[{}]==0 split".format(self._split_index))
            return self._split_0_node._predict_proba(x, omit_warnings, verbose)
        else:
            if verbose:
                self._print("\tpredict_proba: x[{}]==1 split".format(self._split_index))
            return self._split_1_node._predict_proba(x, omit_warnings, verbose)


class ClassicalTreeBase(Node):
    """Classical decision tree base.
    
    Parameters
    ----------
    
    tree_dict : dict
        Tree structure as dictionary, see ``qtree.QTree``.
        
    y_dim : int or None
        Number of binary labels. If None, extract from ``tree_dict``.
        
    y : array-like of shape (n_samples, n_labels) or None
        Target labels of training data. Used to find majority classes for 
        default predictions. If None, no defaults are used.
   
    tree_dict_detailed : dict or None
        Tree structure as detailed dictionary, see ``qtree.QTree``. Is used to 
        add optional information to each node. Set to None to disable.
        
    logger : logging.Logger or None
        Logger for all output purposes. If None, print to std is used.
    """

    def __init__(self, tree_dict, y_dim, y, tree_dict_detailed, logger):
        # y dimension
        y_dim_, found = self._get_y_dim(tree_dict)
        if not found:
            raise ValueError('tree_dict has wrong format')
        if y_dim is not None:
            assert y_dim == y_dim_, 'y_dim and tree_dict incompatible'
        else:
            y_dim = y_dim_

        # y data
        if y is None:
            default_y = None
        else:
            # to determine default_py0_shift (for default prediction based on training data support if no sampling is present in a leaf)
            y = np.asarray(y)
            assert y.shape[1] == y_dim, 'data and tree_dict incompatible'
            self._y0 = np.sum(y, axis=0)
            self._y1 = y.shape[0] - np.sum(y, axis=0)
            default_y = np.empty(y_dim, dtype=int)
            for yi in range(y_dim):
                if self._y0[yi] > self._y1[yi]:
                    default_y[yi] = 1
                elif self._y0[yi] < self._y1[yi]:
                    default_y[yi] = 0
                else:
                    default_y[yi] = None

                    # construct tree
        super().__init__(depth=0, y_dim=y_dim, default_y=default_y, logger=logger)
        self.from_tree_dict(tree_dict, tree_dict_detailed)

    def _get_y_dim(self, value):
        if type(value) is list:
            for value_ in value:
                y_dim, found = self._get_y_dim(value_)
                if found:
                    return y_dim, True
        elif type(value) is dict:
            if len(value) == 0:
                return None, False
            elif all([type(value) is list and all([type(v) is float or type(v) is int for v in value]) for value in
                      value.values()]):
                y_dim = len(list(value.values())[0])
                return y_dim, True
            else:
                for value_ in value.values():
                    y_dim, found = self._get_y_dim(value_)
                    if found:
                        return y_dim, True
        else:
            raise ValueError

    def _tree_dict_traversal(self, tree_dict, paths, path=[]):
        if len(tree_dict) > 0 and type(list(tree_dict.values())[0]) is list and np.all(
                [type(split) is dict for split in list(tree_dict.values())[0]]):
            split_index = list(tree_dict.keys())[0]
            split_list = list(tree_dict.values())[0]
            self._tree_dict_traversal(split_list[0], paths, path + [[split_index, 0]])
            self._tree_dict_traversal(split_list[1], paths, path + [[split_index, 1]])
        else:
            paths.append(path)


class ClassicalTree(ClassicalTreeBase):
    """Classical decision tree base.
    
    Parameters
    ----------
    
    tree_dict : dict
        Tree structure as dictionary, see ``qtree.QTree``.
        
    y_dim : int or None
        Number of binary labels. If None, extract from ``tree_dict``. Defaults 
        to None.
        
    y : array-like of shape (n_samples, n_labels) or None
        Target labels of training data. Used to find majority classes for 
        default predictions. If None, no defaults are used. Defaults to None.
   
    tree_dict_detailed : dict or None
        Tree structure as detailed dictionary, see ``qtree.QTree``. Is used to 
        add optional information to each node. Set to None to disable. Defaults 
        to empty dictionary.
        
    logger : logging.Logger or None
        Logger for all output purposes. If None, print to std is used. Defaults 
        to None.
    """

    def __init__(self, tree_dict, y_dim=None, y=None, tree_dict_detailed={}, logger=None):
        super().__init__(tree_dict=tree_dict, y_dim=y_dim, y=y, tree_dict_detailed=tree_dict_detailed, logger=logger)

    def from_tree_dict(self, tree_dict, tree_dict_detailed={}):
        self._tree_dict = copy.deepcopy(tree_dict)
        if tree_dict_detailed is None:
            tree_dict_detailed = {}
        self._tree_dict_detailed = copy.deepcopy(
            tree_dict_detailed)  # additional information, not actually used for classifier
        self._split(self._tree_dict)
        return self

    def predict(self, X, omit_warnings=False, verbose=False):
        """Make class prediction.
        """

        X = np.asarray(X)
        y = np.empty((X.shape[0], self._y_dim), dtype=bool)
        for idx in range(X.shape[0]):
            y[idx, :] = self._predict(X[idx, :], omit_warnings, verbose)
        return y

    def predict_proba(self, X, omit_warnings=False, verbose=False):
        """Make class probability prediction.
        """

        X = np.asarray(X)
        p = np.empty((X.shape[0], 2, self._y_dim), dtype=np.double)
        for idx in range(X.shape[0]):
            p[idx, :, :] = np.array(self._predict_proba(X[idx, :], omit_warnings, verbose))  # .reshape(2, self._y_dim)
        return p

    def print_classification_report(self, X, y, omit_warnings=False):
        """Print classification report.
        """

        y_predict = self.predict(X, omit_warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(metrics.classification_report(y, y_predict))  # true, pred
        return metrics.accuracy_score(y, y_predict)

    def get_classification_dict(self, X, y, omit_warnings=False):
        """Return classification report as dictionary.
        """

        y_predict = self.predict(X, omit_warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        return metrics.classification_report(y, y_predict, output_dict=True)  # true, pred

    def calc_exact_probabilities(self, X, y, omit_warnings=False):
        """Calculate the exact class prediction probabilities based on the 
        data ``X`` and ``y``.
        """

        # exact probabilities based on data
        paths = []
        self._tree_dict_traversal(self._tree_dict, paths)

        # exact tree dict
        tree_dict_exact = {}  # p(y)
        for path in paths:
            X_filter = X.copy()
            y_filter = y.copy()
            node = tree_dict_exact
            for split_idx, split_value in path:
                y_filter = y_filter[X_filter[:, split_idx] == split_value, :]
                X_filter = X_filter[X_filter[:, split_idx] == split_value, :]
                if split_idx not in node:
                    node[split_idx] = [{}, {}]
                node = node[split_idx][split_value]
            if y_filter.shape[0] > 0:
                py0 = [y_filter[y_filter[:, y_idx] == 0, y_idx].shape[0] / y_filter.shape[0] for y_idx in
                       range(y_filter.shape[1])]
                py1 = [y_filter[y_filter[:, y_idx] == 1, y_idx].shape[0] / y_filter.shape[0] for y_idx in
                       range(y_filter.shape[1])]
            else:  # fall back to unknown
                self._push_warning("Empty y_filter for path = {}".format(path), omit_warnings)
                py0 = [.5] * self._y_dim
                py1 = [.5] * self._y_dim
            node[0] = py0
            node[1] = py1

        def to_data_probabilities(x):
            N = x.shape[1]
            data_probabilities = {}
            for key in product('01', repeat=N):
                data_probabilities["".join(key)] = 0
            for key in product('01', repeat=N):
                key = "".join(key)
                values = [bool(int(value)) for value in key]
                if x.shape[0] > 0:
                    p = x[np.logical_and.reduce([x[:, index] == value for index, value in enumerate(values)])].shape[
                            0] / x.shape[0]
                else:
                    self._push_warning("Empty x for path = {}".format(path), omit_warnings)
                    p = .5
                key = key[::-1]  # correct qiskit ordering
                data_probabilities[key] = p
            return data_probabilities

        # exact conditional probabilities
        tree_dict_exact_data_probabilities = {}
        for path in paths:
            X_filter = X.copy()
            y_filter = y.copy()
            node = tree_dict_exact_data_probabilities
            split_idx_list = []
            for split_idx, split_value in path:
                y_filter = y_filter[X_filter[:, split_idx] == split_value, :]
                X_filter = X_filter[X_filter[:, split_idx] == split_value, :]
                split_idx_list.append(split_idx)
                if split_idx not in node:
                    node[split_idx] = [{}, {}]
                node = node[split_idx][split_value]
            X_filter = np.delete(X_filter, split_idx_list,
                                 axis=1)  # enable: save only reduced, disable: save everything
            Xy_filter = np.concatenate((X_filter, y_filter), axis=1)
            data_probabilities_X = to_data_probabilities(X_filter)
            node['p(x|cl)'] = data_probabilities_X
            data_probabilities_y = to_data_probabilities(y_filter)
            node['p(y|cl)'] = data_probabilities_y
            data_probabilities_Xy = to_data_probabilities(Xy_filter)
            node['p(x,y|cl)'] = data_probabilities_Xy
            p_cl = X_filter.shape[0] / X.shape[0]
            node['p(cl)'] = p_cl

        return paths, tree_dict_exact, tree_dict_exact_data_probabilities

    def calc_difference_to_exact(self, paths, tree_dict_exact, tree_dict_exact_data_probabilities={}, N=None,
                                 epsilon=None, omit_warnings=False):
        """Calculate the distance between the class probabilities in the 
        leaves between the tree dictionary and another tree dictionary.
        """

        # distance between p(y) from two tree_dicts
        tree_dict = self._tree_dict
        tree_dict_detailed = self._tree_dict_detailed

        def p_distance(p_q, p_e):
            p_q = np.array(p_q)  # 1-dim
            p_e = np.array(p_e)  # 1-dim
            # from scipy.spatial.distance import jensenshannon
            # return jensenshannon(p_q, p_e)
            return np.linalg.norm(p_q - p_e)

        # 'p(y)'
        py_dist_list = []
        for path in paths:
            node_q = tree_dict
            node_e = tree_dict_exact
            for split_idx, split_value in path:
                node_q = node_q[split_idx][split_value]
                node_e = node_e[split_idx][split_value]
            if 0 in node_q and 1 in node_q:
                p0_q = node_q[0]
                p1_q = node_q[1]
            else:  # fall back to unknown
                self._push_warning("0, 1 not keys in node_q for path = {}".format(path), omit_warnings)
                p0_q = [.5] * self._y_dim
                p1_q = [.5] * self._y_dim
            p_q = np.array([p0_q, p1_q])[0, :]  # (2, y_dim) -> (y_dim)
            if 0 in node_e and 1 in node_e:
                p0_e = node_e[0]
                p1_e = node_e[1]
            else:  # fall back to unknown
                self._push_warning("0, 1 not keys in node_e for path = {}".format(path), omit_warnings)
                p0_e = [.5] * self._y_dim
                p1_e = [.5] * self._y_dim
            p_e = np.array([p0_e, p1_e])[0, :]  # (2, y_dim) -> (y_dim)
            d = p_distance(p_q, p_e)
            py_dist_list.append(d)
        py_dist_list = np.array(py_dist_list)
        py_dist_mean = np.mean(py_dist_list)

        # 'p(cl)'
        pcl_dist_list = []
        if len(tree_dict_detailed) > 0 and len(tree_dict_exact_data_probabilities) > 0:
            for path in paths:
                node_q = tree_dict_detailed
                node_e = tree_dict_exact_data_probabilities
                for split_idx, split_value in path:
                    node_q = node_q[split_idx][split_value]
                    node_e = node_e[split_idx][split_value]
                if 'p(cl)' in node_q:
                    p_q = node_q['p(cl)']
                else:  # fall back to unknown
                    self._push_warning("'p(cl)' not key in node_q for path = {}".format(path), omit_warnings)
                    p_q = 0
                if 'p(cl)' in node_e:
                    p_e = node_e['p(cl)']
                else:  # fall back to unknown
                    self._push_warning("'p(cl)' not key in node_e for path = {}".format(path), omit_warnings)
                    p_e = 0
                d = p_distance(p_q, p_e)
                pcl_dist_list.append(d)
            pcl_dist_mean = np.mean(pcl_dist_list)
        else:
            pcl_dist_mean = np.nan
        pcl_dist_list = np.array(pcl_dist_list)

        return py_dist_list, py_dist_mean, pcl_dist_list, pcl_dist_mean
