#include <utility>

//
// Created by polarbabe on 05.05.19.
//

#include "Sigmoid.h"

void Sigmoid::compute() {
    x->compute();
    mValue = 1 / (1 + (-(x->mValue)).array().exp());
}

void Sigmoid::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&  g) {
    ComputationalNode::compute_gradients(g);
    auto t = mValue.cwiseProduct(Eigen::MatrixXf::Ones(mValue.rows(), mValue.cols()) - mValue).cwiseProduct(g);
    x->compute_gradients(t);
}

Sigmoid::Sigmoid(std::shared_ptr<ComputationalNode> x) : x(std::move(x)) {

}
