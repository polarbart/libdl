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
    Sum(std::optional<std::shared_ptr<CNode<D, K>>> a, std::shared_ptr<Tensor<D, 0>> r, const std::array<long, K> &shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), std::array<long, 0> {}, r), a(a), shape(shape) {}

    static std::shared_ptr<Tensor<D, 0>> sum(std::shared_ptr<Tensor<D, K>> a) {
        auto result = std::make_shared<Tensor<D, 0>>(a->eTensor->sum(), std::array<long, 0> {});
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Sum<D, K>>(a->gradFn, result, a->eTensor->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<int, K> r;
            r.fill(1);
            a.value()->addGrad(CNode<D, 0>::grad->reshape(r).broadcast(shape));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, K>>> a;
    std::array<long, K> shape;
};

#endif //LIBDL_SUM_H
