##########################################
# File: CMakeLists.txt                   #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(QPBO-ALPHA-EXPAND)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(EIGEN_INCLUDE_DIR "" CACHE PATH "")
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

SET(QPBO_ALPHA_EXPAND_DIR include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/${QPBO_ALPHA_EXPAND_DIR})
SET(QPBO_ALPHA_EXPAND_HDRS ${QPBO_ALPHA_EXPAND_DIR}/qpbo_alpha_expand.h)

SET(QPBO_DIR external/QPBO-v1.32.src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/${QPBO_DIR})
SET(QPBO_SRCS ${QPBO_DIR}/QPBO.cpp
              ${QPBO_DIR}/QPBO_extra.cpp
              ${QPBO_DIR}/QPBO_maxflow.cpp
              ${QPBO_DIR}/QPBO_postprocessing.cpp)
SET(QPBO_HDRS ${QPBO_DIR}/QPBO.h)

ADD_EXECUTABLE(example
               example.cpp
               ${QPBO_ALPHA_EXPAND_HDRS}
               ${QPBO_HDRS}
               ${QPBO_SRCS})
