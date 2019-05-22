//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_SUM_H
#define LIBDL_SUM_H


#include "Tensor.h"
#include "Utils.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename D, int K>
class Sum : public CNode<D, 0> {
public:
    Sum(std::optional<std::shared_ptr<CNode<D, K>>> a, std::shared_ptr<Tensor<D, 0>> r, std::array<long, K> shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), shape(shape) {}

    static std::shared_ptr<Tensor<D, 0>> sum(std::shared_ptr<Tensor<D, K>> a) {
        std::array<long, 0> shape {};
        std::array<long, K> oldShape {};
        std::copy_n(a->eTensor.dimensions().begin(), K, oldShape.begin());
        auto data = std::shared_ptr<D[]>(new D[1]);
        Eigen::TensorMap<Eigen::Tensor<D, 0>> t(data.get(), shape);
        t = a->eTensor.sum();
        auto result = std::make_shared<Tensor<D, 0>>(data, shape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Sum<D, K>>(a->gradFn, result, oldShape));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<int, K> r;
            r.fill(1);
            a.value()->addGrad(CNode<D, 0>::grad.reshape(r).broadcast(shape));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, K>>> a;
    std::array<long, K> shape;
};

#endif //LIBDL_SUM_H
