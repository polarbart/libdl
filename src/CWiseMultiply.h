//
// Created by polarbabe on 08.05.19.
//

#ifndef LIBDL_CWISEMULTIPLY_H
#define LIBDL_CWISEMULTIPLY_H


#include "ComputationalNode.h"

class CWiseMultiply : public ComputationalNode {
public:
    CWiseMultiply(std::shared_ptr<ComputationalNode> a, std::shared_ptr<ComputationalNode> b);
    void compute() override;
    void compute_gradients(const Eigen::Ref<const Eigen::MatrixXf> &param) override;
private:
    std::shared_ptr<ComputationalNode> a;
    std::shared_ptr<ComputationalNode> b;
};


#endif //LIBDL_CWISEMULTIPLY_H
