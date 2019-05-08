//
// Created by polarbabe on 08.05.19.
//

#ifndef LIBDL_REDUCESUM_H
#define LIBDL_REDUCESUM_H


#include "ComputationalNode.h"

class ReduceSum : public ComputationalNode {
public:
    explicit ReduceSum(std::shared_ptr<ComputationalNode> x);

    void compute() override;
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf> &param) override;

private:
    std::shared_ptr<ComputationalNode> x;
};


#endif //LIBDL_REDUCESUM_H
