//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_POW_H
#define LIBDL_POW_H

#include "Tensor.h"
#include "Utils.h"

template <typename D, int R>
class Pow : public CNode<D, R> {
public:
    Pow(std::shared_ptr<Tensor<D, R>> a, const std::array<long, R> &shape, float p)
    : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn}), shape, a), a(a), p(p) {}

    static std::shared_ptr<Tensor<D, R>> pow(std::shared_ptr<Tensor<D, R>> a, float p) {
        auto result = std::make_shared<Tensor<D, R>>(a->eTensor->pow(p), a->eTensor->dimensions());
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Pow<D, R>>(a, a->eTensor->dimensions(), p));
        return result;
    }

    void computeGradients() override {
        if (a->gradFn.has_value())
            a->gradFn.value()->addGrad(a->eTensor->constant(p) * a->eTensor->pow(p - 1) * *CNode<D, R>::grad);
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Tensor<D, R>> a;
    float p;
};


#endif //LIBDL_POW_H
