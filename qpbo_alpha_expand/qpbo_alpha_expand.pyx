##########################################
# File: qpbo_alpha_expand.pyx            #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
cimport numpy as np
np.import_array()

from scipy import sparse

# Requires `rscommon`.
from rscommon.argcheck import check_ndarray_or_raise, check_type_or_raise

# Type
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

# qpbo_alpha_expand_pyx.h
cdef extern from 'qpbo_alpha_expand_pyx.h':
    object qpbo_alpha_expand_ 'qpbo_alpha_expand' (
        np.ndarray,
        list,
        object,
        list,
        np.ndarray,
        np.ndarray,
        np.int32_t)

# check_in_interval_or_raise
def check_in_interval_or_raise(name, a, l, u):
    if not np.all((l <= a) & (a < u)):
        raise ValueError('elements of %s not in [%d, %d)' %
                         (name, l, u))

# qpbo_alpha_expand
def qpbo_alpha_expand(np.ndarray[DTYPE_t, ndim=2, mode='c'] U,
                      list Ps,
                      object G,
                      list valid_labels,
                      np.ndarray[np.int32_t, ndim=1, mode='c']
                        indices_in_valid_labels,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] initial_labels,
                      np.int32_t max_num_iterations):
    """Perform alpha-expansion moves using QPBO.

    Parameters
    ----------
    U : np.ndarray of shape = (num_nodes, num_labels)
        The matrix of unary energies, where `U[i, j]` is the energy of node
        `i` taking label `j`.

    Ps : list of np.ndarray, each of shape = (num_labels, num_labels)
        The list of matrices of pairwise energies, where `P[k][i, j]` is the
        energy of adjacent nodes taking labels `i` and `j`, according to the
        matrix `k`.

    G : scipy.sparse.csr_matrix of shape = (num_nodes, num_nodes)
        If `G[i, j]` is non-zero, then `i` and `j` are adjacent, and the
        matrix `P[G[i, j] - 1]` gives the pairwise energy.

    valid_labels : list of np.ndarray
        Each array in `valid_labels` is a sorted list of label indices.

    indices_in_valid_labels : np.ndarray of shape = (num_nodes,)
        For node `i`, the valid labels are given by
        `valid_labels[indices_in_valid_labels[i]]`.

    initial_labels : np.ndarray of shape = (num_nodes,)
        The array of initial labels which will be refined using
        alpha-expansion QPBO moves.

    max_num_iterations : int
        The maximum number of outer iterations.

    Returns
    -------
    e : float
        The final energy.

    l : np.ndarray of shape = (num_nodes,)
        The array of refined labels corresponding to `e`.
    """
    num_nodes = U.shape[0]
    num_labels = U.shape[1]

    for i, P in enumerate(Ps):
        check_ndarray_or_raise('Ps[%d]' % i, P, DTYPE,
                               2, (num_labels, num_labels),
                               'c_contiguous')

    check_type_or_raise('G', G, sparse.csr_matrix)
    if G.shape != (num_nodes, num_nodes):
        raise ValueError('G.shape != (%d, %d)' % (num_nodes, num_nodes))
    check_ndarray_or_raise('G.data', G.data, np.int32, 1, None)
    check_in_interval_or_raise('G.data', G.data, 1, len(Ps) + 1)

    sorted_valid_labels = []
    for i, labels in enumerate(valid_labels):
        name = 'valid_labels[%d]' % i
        check_ndarray_or_raise(name, labels, np.int32, 1, None,
                               'c_contiguous')
        check_in_interval_or_raise(name, labels, 0, num_labels)
        sorted_valid_labels.append(np.sort(labels))

    check_ndarray_or_raise('indices_in_valid_labels', indices_in_valid_labels,
                           np.int32, 1, (num_nodes,))
    check_in_interval_or_raise('indices_in_valid_labels',
                               indices_in_valid_labels,
                               0, len(sorted_valid_labels))

    check_ndarray_or_raise('initial_labels', initial_labels,
                           np.int32, 1, (num_nodes,))
    check_in_interval_or_raise('initial_labels', initial_labels, 0, num_labels)

    # Solve.
    l = initial_labels.copy()
    e = qpbo_alpha_expand_(U.T, map(np.transpose, Ps), G,
                           sorted_valid_labels, indices_in_valid_labels,
                           l,
                           max_num_iterations)

    return e, l
