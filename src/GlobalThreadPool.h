
#ifndef LIBDL_GLOBALTHREADPOOL_H
#define LIBDL_GLOBALTHREADPOOL_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>


/*
 * Eigen ThreadPools that can be used globally
 * */

class GlobalThreadPool {
public:
    static Eigen::ThreadPool pool;
    static Eigen::ThreadPoolDevice myDevice;
};


#endif //LIBDL_GLOBALTHREADPOOL_H
