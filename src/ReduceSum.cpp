#include <utility>

//
// Created by polarbabe on 08.05.19.
//

#include "ReduceSum.h"

void ReduceSum::compute() {
    x->compute();
    mValue = Eigen::MatrixXf::Constant(1, 1, x->mValue.sum());
}

void ReduceSum::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf> &g) {
    ComputationalNode::compute_gradients(g);
    x->compute_gradients(Eigen::MatrixXf::Constant(x->mValue.rows(), x->mValue.cols(), g.data()[0]));
}

ReduceSum::ReduceSum(std::shared_ptr<ComputationalNode> x) : x(std::move(x)) {

}
