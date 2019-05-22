//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_CNODE_H
#define LIBDL_CNODE_H

#include <memory>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CNodeBase.h"
#include "Tensor.h"


template <typename D, int R>
class CNode : public CNodeBase {
public:
    void addGrad(const Eigen::Tensor<D, R>& g) {
        if (resetGrad) {
            grad = g;
            resetGrad = false;
        } else
            grad += g;
    }
    Eigen::Tensor<D, R> grad;

protected:


    CNode(const std::vector<std::shared_ptr<CNodeBase>>& p, std::weak_ptr<Tensor<D, R>> t) : CNodeBase(p), t(t) {}

    void finishComputeGradient() {
        if (t.expired())
            return;
        auto p = t.lock();
        if (p->requires_grad)
            p->addGrad(grad);
        p->gradFn = std::nullopt;
    }

    std::weak_ptr<Tensor<D, R>> t;
    bool resetGrad = true;
};


#endif //LIBDL_CNODE_H
