//
// Created by polarbabe on 04.05.19.
//

#include <utility>
#include "Variable.h"

Variable::Variable(const Eigen::Ref<const Eigen::MatrixXf>& v) : ComputationalNode(v) {}

void Variable::compute() {

}

void Variable::compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&  t) {
    ComputationalNode::compute_gradients(t);
}
