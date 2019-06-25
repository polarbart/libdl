#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include <unsupported/Eigen/CXX11/ThreadPool>
/*
#include "src/Tensor.h"
#include "src/Sigmoid.h"
#include "src/Add.h"
#include "src/Pow.h"
#include "src/Sub.h"
#include "src/Sum.h"
#include "src/MatMul.h"
*/
using namespace std;
#define R 4
#define D float

Eigen::ThreadPool pool(8);
Eigen::ThreadPoolDevice myDevice(&pool, 8);

template <typename OtherDerived, typename OtherDerived2>
auto myConvolution(const OtherDerived &a, const OtherDerived2 &filter, int padding) {
    Eigen::array<Eigen::IndexPair<int>, R> ePadding{
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(0, 0),
            Eigen::IndexPair<int>(0, 0)
    };
    //auto padded = a.pad(ePadding);
    auto padded = a.pad(ePadding).eval();

    Eigen::array<ptrdiff_t, R> patchDims{filter.dimension(0), filter.dimension(1), a.dimension(2), a.dimension(3)};
    auto patches = padded.extract_patches(patchDims);

    int newHeight = a.dimension(0) - filter.dimension(0) + 2 * padding + 1;
    int newWidth = a.dimension(1) - filter.dimension(1) + 2 * padding + 1;
    Eigen::array<long, R - 1> reshape{patchDims[0] * patchDims[1] * patchDims[2], patchDims[3],
                                      newHeight * newWidth};

    auto im2col = patches.reshape(reshape);
    auto i2cFilter = filter.reshape(Eigen::array<long, 2>{reshape[0], filter.dimension(3)});

    auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});

    Eigen::array<int, R - 1> shuffle{2, 0, 1};
    Eigen::array<long, R> reshape2{newHeight, newWidth, filter.dimension(3), a.dimension(3)};
    return conv.shuffle(shuffle).eval().reshape(reshape2);
}

int main() {
    Eigen::Tensor<float, 4> t(14,14,32,64);
    t.setRandom();
    Eigen::Tensor<float, 4> f(5,5,32,64);
    f.setRandom();
    Eigen::Tensor<float, 1> b(64);
    b.setRandom();

    Eigen::array<long, R> reshape {1, 1, 64, 1};
    Eigen::array<long, R> broadcast {14, 14, 1, 64};



    Eigen::Tensor<float, 4> r(14,14,64,64);
    auto begin = std::chrono::high_resolution_clock::now();
    r.device(myDevice) = myConvolution(t, f, 2) + b.reshape(reshape).broadcast(broadcast);

    std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - begin;
    std::cout << elapsed_seconds.count() << std::endl;
    std::cout << r(0, 0, 0, 0) << std::endl;

}