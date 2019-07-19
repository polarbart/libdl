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
    explicit Leaf(const std::shared_ptr<Tensor<D, R>> &t) : CNode<D, R>(std::vector<std::shared_ptr<CNodeBase>> {}, t) {}
    void computeGradients() override {

        CNode<D, R>::resetGrad = true;
        if (CNode<D, R>::holder.expired())
            return;
        auto p = CNode<D, R>::holder.lock();
        if (p->requiresGrad)
            p->addGrad(CNode<D, R>::grad);
    }

};


#endif //LIBDL_LEAF_H
