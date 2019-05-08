//
// Created by polarbabe on 04.05.19.
//

#include "Multiply.h"

void Multiply::compute() {
    a->compute();
    b->compute();
    mValue = a->mValue * b->mValue;
}

void Multiply::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>& g) {
    ComputationalNode::compute_gradients(g);
    auto ga = g * b->mValue.transpose();
    auto gb = a->mValue.transpose() * g;
    a->compute_gradients(ga);
    b->compute_gradients(gb);
}

Multiply::Multiply(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b)
        : a(std::move(a)), b(std::move(b)) {}
