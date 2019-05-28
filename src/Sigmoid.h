//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_SIGMOID_H
#define LIBDL_SIGMOID_H

#include "Tensor.h"
#include "Utils.h"

template <typename D, int R>
class Sigmoid : public CNode<D, R> {
public:
    Sigmoid(const std::optional<std::shared_ptr<CNode<D, R>>> &a, const std::shared_ptr<Tensor<D, R>> &r)
    : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), r(r->eTensor) {}

    static std::shared_ptr<Tensor<D, R>> sigmoid(const std::shared_ptr<Tensor<D, R>> &a) {
        auto x = a->eTensor->constant(1) / (a->eTensor->constant(1) + (-(*a->eTensor)).exp());
        auto result = std::make_shared<Tensor<D, R>>(x, a->eTensor->dimensions());
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Sigmoid<D, R>>(a->gradFn, result));
        return result;
    }

    void computeGradients() override {
        if (a.has_value())
            a.value()->addGrad(*r * (r->constant(1) - *r) * *CNode<D, R>::grad);
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> a;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> r;
};


#endif //LIBDL_SIGMOID_H
