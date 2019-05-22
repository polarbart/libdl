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
    Sigmoid(std::optional<std::shared_ptr<CNode<D, R>>> a, std::shared_ptr<Tensor<D, R>> r)
    : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), r(r) {}

    static std::shared_ptr<Tensor<D, R>> sigmoid(std::shared_ptr<Tensor<D, R>> a) {
        std::array<long, R> shape {};
        std::copy_n(a->eTensor.dimensions().begin(), R, shape.begin());
        auto data = std::shared_ptr<D[]>(new D[a->eTensor.size()]);
        Eigen::TensorMap<Eigen::Tensor<D, R>> t(data.get(), shape);
        t = a->eTensor.constant(1) / (a->eTensor.constant(1) + (-a->eTensor).exp());
        auto result = std::make_shared<Tensor<D, R>>(data, shape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Sigmoid<D, R>>(a->gradFn, result));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            Eigen::Tensor<D, R> t = r->eTensor * (r->eTensor.constant(1) - r->eTensor) * CNode<D, R>::grad;
            a.value()->addGrad(t);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> a;
    std::shared_ptr<Tensor<D, R>> r;
};


#endif //LIBDL_SIGMOID_H
