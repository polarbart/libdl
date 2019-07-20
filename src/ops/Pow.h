//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_POW_H
#define LIBDL_POW_H

#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int R>
class Pow : public CNode<D, R> {
public:
    Pow(const std::shared_ptr<Tensor<D, R>> &a, float p, const std::shared_ptr<Tensor<D, R>> &t)
    : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn}), t), a(a->eTensor), ca(a->gradFn), p(p) {}

    static std::shared_ptr<Tensor<D, R>> pow(const std::shared_ptr<Tensor<D, R>> &a, float p) {
        auto result = std::make_shared<Tensor<D, R>>(a->eTensor->pow(p), a->eTensor->dimensions());
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Pow<D, R>>(a, p, result));
        return result;
    }

    void computeGradients() override {
        if (ca.has_value())
            ca.value()->addGrad(a->constant(p) * a->pow(p - 1) * *CNode<D, R>::grad);
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> a;
    std::optional<std::shared_ptr<CNode<D, R>>> ca;
    float p;
};


#endif //LIBDL_POW_H
