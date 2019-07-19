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
#define INT_CEIL(x, y) ((x)/(y) + ((x) % (y) != 0))


void dialate(Eigen::Tensor<float, 2> &d, Eigen::Tensor<float, 2> &m, int stride) {
    d.setConstant(0);
    for (int b = 0, bm = 0; b < d.dimension(1); b += stride, bm++) {
        for (int a = 0, am = 0; a < d.dimension(0); a += stride, am++) {
            d(a, b) = m(am, bm);
        }
    }
}

int main() {
    Eigen::Tensor<float, 3> m(1, 32, 32);
    for (int i = 0; i < m.size(); i++)
        m.data()[i] = i+1;
    std::cout << m.chip(0, 0) << std::endl;
    //Eigen::Tensor<float, 4> r = m.extract_image_patches(2, 2, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 0);
    Eigen::Tensor<float, 4> r = m.extract_image_patches(5, 5, 1,1,1,1,2,2,2,3,2,3,0);
    std::cout << r.dimension(3) << std::endl;
    for (int i = 0; i < r.dimension(3); i++)
        std::cout << r.chip(i, 3).chip(0, 0) << std::endl<<std::endl;
}

/*
 extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride,
                          const Index row_inflate_stride, const Index col_inflate_stride,
                          const Index padding_top, const Index padding_bottom,
                          const Index padding_left,const Index padding_right,
                          const Scalar padding_value)


extract_image_patches(const Index patch_rows = 1, const Index patch_cols = 1,
                          const Index row_stride = 1, const Index col_stride = 1,
                          const Index in_row_stride = 1, const Index in_col_stride = 1,
                          const PaddingType padding_type = PADDING_SAME, const Scalar padding_value = Scalar(0)) const {
 */