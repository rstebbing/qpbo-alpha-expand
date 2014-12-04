##########################################
# File: example.py                       #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
from scipy import sparse
from scipy.spatial import distance

from qpbo_alpha_expand import qpbo_alpha_expand

# main
def main():
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0]], dtype=np.float64)

    # Refer to qpbo_alpha_expand_example.cpp.
    np.random.seed(1337)
    Y = 0.10 * np.random.randn(32 * 2).reshape(32, 2)
    Y[16:] += X[1]

    U = 2.0 * distance.cdist(X, Y, 'sqeuclidean')
    P = distance.cdist(Y, Y, 'sqeuclidean')

    G = sparse.lil_matrix((2, 2), dtype=np.int32)
    G[0, 1] = 1
    G = G.tocsr()

    l = np.array([0, 0], dtype=np.int32)
    print 'Initial `l`:', l

    e, l = qpbo_alpha_expand(
        U, [P], G,
        [np.arange(Y.shape[0], dtype=np.int32)], # `valid_labels`.
        np.zeros(2, dtype=np.int32), # `indices_in_valid_labels`.
        np.zeros(2, dtype=np.int32), # `initial_labels`.
        100)

    print 'Final `l`:', l
    print 'Energy (returned):', e
    l0, l1 = l
    print 'Energy (manual):', U[0, l0] + U[1, l1] + P[l0, l1]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    f, ax = plt.subplots()
    x, y = Y.T
    ax.plot(x, y, 'r.')
    x, y = X.T
    ax.plot(x, y, 'bo')
    x, y = Y[l].T
    ax.plot(x, y, 'm^')
    plt.show()

if __name__ == '__main__':
    main()
