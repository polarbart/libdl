//
// Created by polarbabe on 04.05.19.
//

#ifndef LIBDL_MULTIPLY_H
#define LIBDL_MULTIPLY_H

#include "ComputationalNode.h"

class Multiply : public ComputationalNode {

public:
    Multiply(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b);

    void compute() override;
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&) override;

private:
    std::shared_ptr<ComputationalNode> a;
    std::shared_ptr<ComputationalNode> b;
};


#endif //LIBDL_MULTIPLY_H
