//
// Created by polarbabe on 04.05.19.
//

#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H


#include "ComputationalNode.h"
#include <memory>
#include <Eigen/Dense>

class Variable : public ComputationalNode {
public:
    explicit Variable(const Eigen::Ref<const Eigen::MatrixXf>&);
    void compute() override;
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>& ) override;
};


#endif //LIBDL_TENSOR_H
