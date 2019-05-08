//
// Created by polarbabe on 04.05.19.
//

#ifndef LIBDL_ADD_H
#define LIBDL_ADD_H


#include "ComputationalNode.h"
#include <memory>

class Add : public ComputationalNode {
public:
    Add(std::shared_ptr<ComputationalNode>, std::shared_ptr<ComputationalNode>);
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&) override;
    void compute() override;

private:
    std::shared_ptr<ComputationalNode> a;
    std::shared_ptr<ComputationalNode> b;
};


#endif //LIBDL_ADD_H
