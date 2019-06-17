//
// Created by superbabes on 16.06.19.
//

#ifndef LIBDL_SOFTMAX_H
#define LIBDL_SOFTMAX_H


#include "CNode.h"
#include "Utils.h"

template <typename D, int R>
class Softmax : public CNode<D, R> {

public:
    Softmax(const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &result) : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result), cx(x->gradFn), r(result) {}

    static std::shared_ptr<Tensor<D, R>> softmax(const std::shared_ptr<Tensor<D, R>> &x) {
        Eigen::array<long, R> reshape;
        std::copy_n(std::begin(x->eTensor->dimensions()), R, std::begin(reshape));
        reshape[R-1] = 1;
        Eigen::array<long, R> broadcast;
        broadcast.fill(1);
        reshape[R-1] = x->eTensor->dimension(R-1);
        auto i1 = (*x->eTensor - x->eTensor->maximum(Eigen::array<int, 1> {R-1}).reshape(reshape).broadcast(broadcast)).exp();
        auto i2 = i1 / i1.sum(Eigen::array<int, 1> {R-1}).reshape(reshape).broadcast(broadcast);
        auto result = std::make_shared<Tensor<D, R>>(i2, x->eTensor->dimensions());
        if (x->needsGradient())
            result->setGradFn(std::make_shared<Softmax<D, R>>(x, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            auto i1 = *r * (r->constant(1) - *r);
            //cx.value()->addGrad(in * *CNode<D, R>::grad);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> r;
};

#endif //LIBDL_SOFTMAX_H
