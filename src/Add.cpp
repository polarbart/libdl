#include <utility>

//
// Created by polarbabe on 04.05.19.
//

#include "Add.h"

Add::Add(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b) : a(std::move(a)), b(std::move(b)) {}

void Add::compute() {
    a->compute();
    b->compute();
    mValue = a->mValue + b->mValue;
}

void Add::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&  g){
    ComputationalNode::compute_gradients(g);
    a->compute_gradients(g);
    b->compute_gradients(g);
}
