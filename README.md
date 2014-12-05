qpbo-alpha-expand
=================

A C++ implementation of alpha-expansion moves using QPBO for minimisation of non-submodular energies.
`qpbo_alpha_expand` is inspired by [Shai Bagon's MATLAB wrapper] [1] and uses [Vladimir Kolmogorov's QPBO implementation] [2].

This project was motivated by the need to minimise problems with large label spaces where direct instantiation of the unary and pairwise energy matrices was not possible.
For the problems of interest, the label spaces for each node were also restricted.
This `qpbo_alpha_expand` implementation makes use of templates to enable straightforward lazy evaluation of the unary and pairwise energies and restriction of the label spaces.

Author: Richard Stebbing

License: MIT (refer to LICENSE).

**As per the license for QPBO, this software can be used for research purposes only.**

Dependencies
------------
* [QPBO] [2]
* [Eigen 3] [3]

Python:
* [rstebbing/common] [4]
* Numpy
* Scipy

C++
---
1. Extract [QPBO-v1.32.src.zip] [2] to the subdirectory external/ under the project root.
All QPBO files should reside under external/QPBO-v1.32.src/.
2. Run CMake with an out of source build.
3. Set `EIGEN_INCLUDE_DIR` to the full path up to and including eigen3/.
(Add `-std=c++11` to `CMAKE_CXX_FLAGS` if compiling with gcc.)
4. Configure.
5. Build.
5. Run [example](example.cpp).

Python
------
1. Extract [QPBO-v1.32.src.zip] [2] as above.
2. Set `EIGEN_INCLUDE` and `COMMON_CPP_INCLUDE` in site.cfg.
3. Build the Python extension in place: `python setup.py build_ext --inplace`.
(Use `export CFLAGS=-std=c++11` beforehand if compiling with gcc.)
4. With [rstebbing/common](https://github.com/rstebbing/common/tree/master) installed, run [example.py](example.py).

To make `qpbo_alpha_expand` available to other projects (either globally or under a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs)):

1. Build: `python setup.py build`.
(Use `export CFLAGS=-std=c++11` beforehand if compiling with gcc.)
2. Install: `python setup.py install`.

[1]: http://www.wisdom.weizmann.ac.il/~bagon/matlab_code/ExtendedGCmex1.3.tar.gz
[2]: http://pub.ist.ac.at/~vnk/software/QPBO-v1.32.src.zip
[3]: http://eigen.tuxfamily.org
[4]: http://github.com/rstebbing/common
