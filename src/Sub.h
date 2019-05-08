//
// Created by polarbabe on 08.05.19.
//

#ifndef LIBDL_SUB_H
#define LIBDL_SUB_H


#include "ComputationalNode.h"

class Sub : public ComputationalNode {

public:
    Sub(std::shared_ptr<ComputationalNode>, std::shared_ptr<ComputationalNode>);
    void compute() override;
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf> &param) override;

private:
    std::shared_ptr<ComputationalNode> a;
    std::shared_ptr<ComputationalNode> b;
};


#endif //LIBDL_SUB_H
