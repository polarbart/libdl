//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_LEAF_H
#define LIBDL_LEAF_H

#include "CNode.h"

template <typename D, int R>
class Leaf : public CNode<D, R> {
public:
    explicit Leaf(std::weak_ptr<Tensor<D, R>> t) : CNode<D, R>(std::vector<std::shared_ptr<CNodeBase>> {}, t) {}
    void computeGradients() override {
        CNode<D, R>::resetGrad = true;
        if (CNode<D, R>::t.expired())
            return;
        auto p = CNode<D, R>::t.lock();
        if (p->requires_grad)
            p->addGrad(CNode<D, R>::grad);
    }

};


#endif //LIBDL_LEAF_H
