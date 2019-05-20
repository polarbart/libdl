//
// Created by polarbabe on 20.05.19.
//

#ifndef LIBDL_TESTTENSOR_H
#define LIBDL_TESTTENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "CNode.h"


class TestTensor {

public:
    explicit TestTensor(Eigen::TensorBase<Eigen::Tensor<float, 1>> t) : t1(t), rank(1), requiresGrad(false) {}
    explicit TestTensor(Eigen::TensorBase<Eigen::Tensor<float, 2>> t) : t2(t), rank(2), requiresGrad(false) {}
    explicit TestTensor(Eigen::TensorBase<Eigen::Tensor<float, 3>> t) : t3(t), rank(3), requiresGrad(false) {}
    explicit TestTensor(Eigen::TensorBase<Eigen::Tensor<float, 4>> t) : t4(t), rank(4), requiresGrad(false) {}

    const int rank;
    bool requiresGrad;

    std::shared_ptr<TestTensor> add(const std::shared_ptr<TestTensor>& b);
    std::shared_ptr<TestTensor> matmul(const std::shared_ptr<TestTensor>& b);
    bool needsGradient();

private:
    const Eigen::Tensor<float, 1> t1;
    const Eigen::Tensor<float, 2> t2;
    const Eigen::Tensor<float, 3> t3;
    const Eigen::Tensor<float, 4> t4;

    std::optional<std::shared_ptr<CNode>> grad_fn;

    void setGradFn(const std::shared_ptr<CNode>&);
};


#endif //LIBDL_TESTTENSOR_H
