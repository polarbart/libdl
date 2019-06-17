//
// Created by polarbabe on 28.05.19.
//

#ifndef LIBDL_LEAKYRELU_H
#define LIBDL_LEAKYRELU_H

#include "CNode.h"
#include "Utils.h"

#define NEGATIVE_SLOPE .01

template <typename D, int R>
class LeakyRelu : public CNode<D, R> {

public:
    LeakyRelu(const std::shared_ptr<Tensor<D, R>> &x,
              const std::shared_ptr<Tensor<D, R>> &result) : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result), x(x->eTensor), cx(x->gradFn) {}

    static std::shared_ptr<Tensor<D, R>> leakyRelu(const std::shared_ptr<Tensor<D, R>> &x) {
        auto mask = (*x->eTensor >= x->eTensor->constant(0)).select(x->eTensor->constant(1), x->eTensor->constant(NEGATIVE_SLOPE));
        auto result = std::make_shared<Tensor<D, R>>(*x->eTensor * mask, x->eTensor->dimensions());
        if (x->needsGradient())
            result->setGradFn(std::make_shared<LeakyRelu<D, R>>(x, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            auto mask = (*x >= x->constant(0)).select(x->constant(1), x->constant(NEGATIVE_SLOPE));
            cx.value()->addGrad(mask * *CNode<D, R>::grad);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;


};

#undef NEGATIVE_SLOPE

#endif //LIBDL_LEAKYRELU_H
