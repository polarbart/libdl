//
// Created by polarbabes on 16.05.19.
//

#ifndef LIBDL_ADD_H
#define LIBDL_ADD_H

#include "Tensor.h"
#include "CNode.h"
#include "Utils.h"

template <typename D, int RA, int RB>
class Add : public CNode<D, std::max(RA, RB)>  {

public:
    Add(const std::optional<std::shared_ptr<CNode<D, RA>>> &a,
        const std::optional<std::shared_ptr<CNode<D, RB>>> &b,
        const std::shared_ptr<Tensor<D, std::max(RA, RB)>> &t)
    : CNode<D, std::max(RA, RB)>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a, b}), t), a(a), b(b) {};

    static std::shared_ptr<Tensor<D, std::max(RA, RB)>> add(const std::shared_ptr<Tensor<D, RA>> &a, const std::shared_ptr<Tensor<D, RB>> &b) {
        if constexpr (RB > RA)
            return Add<D, RB, RA>::add(b, a);
        else {
            std::shared_ptr<Tensor<D, RA>> result;
            if constexpr (RB < RA) {
                std::array<int, RA> reshape{};
                std::array<int, RA> broadcast{};
                for (int i = 0; i < RB; ++i) {
                    reshape[i] = b->eTensor->dimension(i);
                    broadcast[i] = 1;
                }
                for (int i = RB; i < RA; ++i) {
                    reshape[i] = 1;
                    broadcast[i] = a->eTensor->dimension(i);
                }
                result = std::make_shared<Tensor<D, RA>>(*a->eTensor + b->eTensor->reshape(reshape).broadcast(broadcast), a->eTensor->dimensions());
            } else
                result = std::make_shared<Tensor<D, RA>>(*a->eTensor + *b->eTensor, a->eTensor->dimensions());

            if (a->needsGradient() || b->needsGradient())
                result->setGradFn(std::make_shared<Add<D, RA, RB>>(a->gradFn, b->gradFn, result));
            return result;
        }
    }

    void computeGradients() override {
        if (a.has_value())
            a.value()->addGrad(*CNode<D, RA>::grad);
        if (b.has_value()) {
            if constexpr (RB < RA) {
                std::array<int, RA - RB> sum{};
                for (int i = 0; i < RA - RB; ++i)
                    sum[i] = RB + i;
                b.value()->addGrad(CNode<D, RA>::grad->sum(sum));
            } else
                b.value()->addGrad(*CNode<D, RA>::grad);
        }
        CNode<D, RA>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> a;
    std::optional<std::shared_ptr<CNode<D, RB>>> b;
};


#endif //LIBDL_ADD_H
