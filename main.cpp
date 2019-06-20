#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/CXX11/Tensor>
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

int main() {
    int padding = 0;

    Eigen::Tensor<D, 1> t(8);
    t.setValues({0,1,2,3,4,5,6,7});//,8,9,10,11,12,13,14,15});
    Eigen::Tensor<D, R> a = t.reshape(Eigen::array<long, R> {2, 2, 1, 2});
    Eigen::Tensor<D, R> grad = a;

    Eigen::array<Eigen::IndexPair<int>, R> ePadding{
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(0, 0),
            Eigen::IndexPair<int>(0, 0),
    };
    Eigen::Tensor<D, R> padded = a.pad(ePadding);

    Eigen::array<ptrdiff_t, R> patchDims {grad.dimension(0), grad.dimension(1), 1, a.dimension(3)};
    Eigen::Tensor<D, R+1> patches = padded.extract_patches(patchDims);

    Eigen::array<long, R-1> reshape {patchDims[0]*patchDims[1], patchDims[3], a.dimension(2)};
    Eigen::Tensor<D, R-1> im2col = patches.reshape(reshape);
    Eigen::Tensor<D, 2> i2cFilter = grad.sum(Eigen::array<int, 1> {3}).reshape(Eigen::array<long, 2> {reshape[0], grad.dimension(2)});
    Eigen::Tensor<D, R-1> conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(0, 0)});

    Eigen::array<int, R-1> shuffle {2, 0, 1};
    Eigen::array<long, R+1> reshape2 {1,1, a.dimension(2), grad.dimension(2), a.dimension(3)};
    std::cout << conv.shuffle(shuffle).reshape(reshape2).sum(Eigen::array<int, 1> {4}).eval() << std::endl;
}