#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "src/Tensor.h"
#include "src/Sigmoid.h"
#include "src/Add.h"
#include "src/Pow.h"
#include "src/Sub.h"
#include "src/Sum.h"
#include "src/MatMul.h"

using namespace std;

//#define forward(x) Sigmoid<float, 2>::sigmoid(Add<float, 2, 1>::add(Matmul<float, 2, 1>::matmul(Sigmoid<float, 2>::sigmoid(Add<float, 2, 1>::add(Matmul<float, 2, 2>::matmul(x, W1), B1)), W2), B2))

int main() {
    auto W1 = std::make_shared<Tensor<float, 2>>(std::shared_ptr<float[]>(new float[4] {.05, .0009, -.04, -.1}), std::array<long, 2> {2, 2}, true);
    /*auto B1 = std::make_shared<Tensor<float, 1>>(std::shared_ptr<float[]>(new float[2] {0, 0}), std::array<long, 1> {2}, true);
    auto W2 = std::make_shared<Tensor<float, 2>>(std::shared_ptr<float[]>(new float[2] {.051, -.045}), std::array<long, 2> {2, 1}, true);
    auto B2 = std::make_shared<Tensor<float, 1>>(std::shared_ptr<float[]>(new float[1] {0}), std::array<long, 1> {1}, true);

    auto x = std::make_shared<Tensor<float, 2>>(std::shared_ptr<float[]>(new float[8] {0, 0, 1, 1, 0, 1, 0, 1}), std::array<long, 2> {4, 2});
    auto y = std::make_shared<Tensor<float, 2>>(std::shared_ptr<float[]>(new float[4] {0, 1, 1, 0}), std::array<long, 2> {4, 1});

    cout << x << endl;
    cout << y << endl;

*/
}