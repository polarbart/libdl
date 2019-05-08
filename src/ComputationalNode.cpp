//
// Created by polarbabe on 03.05.19.
//

#include "ComputationalNode.h"
#include <utility>

ComputationalNode::ComputationalNode(const Eigen::Ref<const Eigen::MatrixXf>& mValue) : mValue(mValue) {}

void ComputationalNode::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>& g) {
    if (resetGradient) {
        mGradient = g;
        resetGradient = false;
    } else
        mGradient += g;
}

void ComputationalNode::zero_gradient() {
    resetGradient = true;
}

ComputationalNode::ComputationalNode() = default;


