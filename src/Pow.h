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
    Pow(std::shared_ptr<Tensor<D, R>> a, float p)
    : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn}), a), a(a), p(p) {}

    static std::shared_ptr<Tensor<D, R>> pow(std::shared_ptr<Tensor<D, R>> a, float p) {
        std::array<long, R> shape {};
        std::copy_n(a->eTensor.dimensions().begin(), R, shape.begin());
        auto data = std::shared_ptr<D[]>(new D[a->eTensor.size()]);
        Eigen::TensorMap<Eigen::Tensor<D, R>> t(data.get(), shape);
        t = a->eTensor.pow(p);
        auto result = std::make_shared<Tensor<D, R>>(data, shape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Pow<D, R>>(a, p));
        return result;
    }

    void computeGradients() override {
        if (a->gradFn.has_value()) {
            Eigen::Tensor<D, R> t = a->eTensor.constant(p) * a->eTensor.pow(p - 1) * CNode<D, R>::grad;
            a->gradFn.value()->addGrad(t);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Tensor<D, R>> a;
    float p;
};


#endif //LIBDL_POW_H
