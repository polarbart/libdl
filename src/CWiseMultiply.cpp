//
// Created by polarbabe on 08.05.19.
//

#include "CWiseMultiply.h"

void CWiseMultiply::compute() {
    a->compute();
    b->compute();
    mValue = a->mValue.cwiseProduct(b->mValue);
}

void CWiseMultiply::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>& g) {
    ComputationalNode::compute_gradients(g);
    auto t1 = b->mValue.cwiseProduct(g);
    auto t2 = a->mValue.cwiseProduct(g);
    a->compute_gradients(t1);
    b->compute_gradients(t2);
}

CWiseMultiply::CWiseMultiply(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b) : a(std::move(a)), b(std::move(b)) {

}
