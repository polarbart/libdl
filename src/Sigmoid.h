//
// Created by polarbabe on 05.05.19.
//

#ifndef LIBDL_SIGMOID_H
#define LIBDL_SIGMOID_H


#include "ComputationalNode.h"

class Sigmoid : public ComputationalNode {
public:
    void compute() override;

    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>& ) override;

    explicit Sigmoid(std::shared_ptr<ComputationalNode>);

private:
    std::shared_ptr<ComputationalNode> x;
};


#endif //LIBDL_SIGMOID_H
