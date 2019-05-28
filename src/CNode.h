//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_CNODE_H
#define LIBDL_CNODE_H

#include <memory>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CNodeBase.h"
#include "ETensor.h"

template <typename D, int R>
class Tensor;

template <typename D, int R>
class CNode : public CNodeBase {
public:
    template <typename Derived>
    void addGrad(const Derived &g) {
        if (resetGrad) {
            if (grad.use_count() == 0) {
                auto p = t.lock();
                if (!t.expired() && p->grad.use_count() > 0) {
                    grad = p->grad;
                } else
                    grad = std::make_shared<ETensor<D, R>>(g, shape);
            } else
                *grad = g;
            resetGrad = false;
        } else
            *grad += g;
    }

    void zeroGrad() {
        resetGrad = true;
    }

    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> grad;

protected:
    const std::array<long, R> shape;
    CNode(const std::vector<std::shared_ptr<CNodeBase>>& p, const std::shared_ptr<Tensor<D, R>> &t) : CNodeBase(p), shape(t->eTensor->dimensions()), t(t) {}

    void finishComputeGradient() {
        if (t.expired())
            return;
        auto p = t.lock();
        if (p->requiresGrad)
            p->addGrad(grad);
        p->gradFn = std::nullopt;
    }

    std::weak_ptr<Tensor<D, R>> t;
    bool resetGrad = true;
};


#endif //LIBDL_CNODE_H
