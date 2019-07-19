//
// Created by superbabes on 19.07.19.
//

#ifndef LIBDL_GLOBALS_H
#define LIBDL_GLOBALS_H

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>



class Globals {
    static bool forceNoGrad;
    static Eigen::ThreadPool pool;
    static Eigen::ThreadPoolDevice myDevice;
};


#endif //LIBDL_GLOBALS_H
