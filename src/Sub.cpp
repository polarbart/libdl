//
// Created by polarbabe on 08.05.19.
//

#include "Sub.h"

void Sub::compute() {
    a->compute();
    b->compute();
    mValue = a->mValue - b->mValue;
}

void Sub::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf> &g) {
    ComputationalNode::compute_gradients(g);
    a->compute_gradients(g);
    b->compute_gradients(-g);
}

Sub::Sub(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b) : a(std::move(a)), b(std::move(b)) {}
