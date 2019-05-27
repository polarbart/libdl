//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_LEAF_H
#define LIBDL_LEAF_H

#include "CNode.h"

template <typename D, int R>
class Tensor;

template <typename D, int R>
class Leaf : public CNode<D, R> {
public:
    Leaf(std::weak_ptr<Tensor<D, R>> t, const std::array<long, R> &d) : CNode<D, R>(std::vector<std::shared_ptr<CNodeBase>> {}, d, t) {}
    void computeGradients() override {
        CNode<D, R>::resetGrad = true;
        if (CNode<D, R>::t.expired())
            return;
        auto p = CNode<D, R>::t.lock();
        if (p->requiresGrad)
            p->addGrad(CNode<D, R>::grad);
    }

};


#endif //LIBDL_LEAF_H
