//
// Created by polarbabe on 03.05.19.
//

#ifndef LIBDL_COMPUTATIONALNODE_H
#define LIBDL_COMPUTATIONALNODE_H

#include <Eigen/Dense>
#include <memory>

class ComputationalNode {
public:
    virtual void compute() = 0;
    virtual void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf>&);
    void zero_gradient();

    explicit ComputationalNode(const Eigen::Ref<const Eigen::MatrixXf>&);
    ComputationalNode();
    Eigen::MatrixXf mValue;
    Eigen::MatrixXf mGradient;

private:
    bool resetGradient = true;
};


#endif //LIBDL_COMPUTATIONALNODE_H
